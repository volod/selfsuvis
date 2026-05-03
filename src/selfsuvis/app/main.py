import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from selfsuvis.app.db import close_db_pool, init_db_pool
from selfsuvis.app.routers.admin import router as admin_router
from selfsuvis.app.routers.cvat import cvat_admin_router, webhook_router
from selfsuvis.app.routers.health import router as health_router
from selfsuvis.app.routers.index import router as index_router
from selfsuvis.app.routers.jobs import router as jobs_router
from selfsuvis.app.routers.query import router as query_router
from selfsuvis.app.routers.realtime import router as realtime_router
from selfsuvis.app.routers.robot import router as robot_router
from selfsuvis.app.routers.scene import router as scene_router
from selfsuvis.app.routers.site_state import router as site_state_router
from selfsuvis.app.services.coop_streams import CoopStreamService
from selfsuvis.app.services.form_templates import get_index_form_html
from selfsuvis.app.services.live_streams import MediaMtxClient, RealtimeStreamManager
from selfsuvis.pipeline.core import get_logger, log_preflight, run_production_preflight
from selfsuvis.pipeline.storage.processed import ainit_db as init_processed_db

logger = get_logger(__name__)


def _start_coop_pilot(app: FastAPI) -> "asyncio.Task | None":
    """Start the coop_pilot MQTT subscriber, site state aggregator, and threat pipeline."""
    try:
        from selfsuvis.coop_pilot.config import settings as coop_settings
        from selfsuvis.coop_pilot.mesh.fusion import SensorMeshFusion
        from selfsuvis.coop_pilot.mesh.scene_synthesis import SceneSynthesizer
        from selfsuvis.coop_pilot.mesh.site_state import SiteStateAggregator
        from selfsuvis.coop_pilot.sensors.mqtt_subscriber import MqttSubscriber
        from selfsuvis.pipeline.realtime.aggregator import RealtimeThreatAggregator
        from selfsuvis.pipeline.realtime.coop_ingest import CoopRealtimeIngestor

        aggregator = SiteStateAggregator()
        fusion = SensorMeshFusion(aggregator)
        threat_agg = RealtimeThreatAggregator()
        ingestor = CoopRealtimeIngestor(threat_agg)
        synthesizer = SceneSynthesizer(
            aggregator=aggregator,
            db_pool=getattr(app.state, "db_pool", None),
        )

        app.state.site_state_aggregator = aggregator
        app.state.sensor_mesh_fusion = fusion
        app.state.coop_threat_aggregator = threat_agg
        app.state.scene_synthesizer = synthesizer

        subscriber = MqttSubscriber(
            on_sensor=_multi_callback(aggregator.ingest_sensor_reading, ingestor.on_sensor_reading),
            on_camera=_multi_callback(aggregator.ingest_camera_event, ingestor.on_camera_event),
        )
        task = asyncio.create_task(subscriber.run(), name="coop_pilot_mqtt")
        logger.info(
            "coop_pilot started (broker: %s:%d)",
            coop_settings.mqtt_host,
            coop_settings.mqtt_port,
        )
        return task
    except Exception as exc:
        logger.warning("coop_pilot not started: %s", exc)
        return None


def _multi_callback(*cbs):
    """Return an async callback that fans out to all provided callbacks."""
    async def _cb(event) -> None:
        for cb in cbs:
            try:
                await cb(event)
            except Exception:
                pass
    return _cb


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared resources."""
    report = run_production_preflight("api")
    log_preflight(report)
    if os.getenv("STARTUP_PREFLIGHT_STRICT", "false").lower() == "true":
        report.raise_for_errors()
    await init_processed_db()
    await init_db_pool(app)
    app.state.mediamtx_client = MediaMtxClient()
    app.state.realtime_stream_manager = RealtimeStreamManager(getattr(app.state, "db_pool", None))

    mqtt_task = _start_coop_pilot(app)

    coop_streams = CoopStreamService(
        mediamtx_client=app.state.mediamtx_client,
        stream_manager=app.state.realtime_stream_manager,
        site_aggregator=getattr(app.state, "site_state_aggregator", None),
        db_pool=getattr(app.state, "db_pool", None),
    )
    app.state.coop_streams = coop_streams
    await coop_streams.start()

    try:
        yield
    finally:
        if mqtt_task and not mqtt_task.done():
            mqtt_task.cancel()
            await asyncio.gather(mqtt_task, return_exceptions=True)
        await coop_streams.shutdown()
        await app.state.realtime_stream_manager.shutdown()
        await close_db_pool(app)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


app.include_router(admin_router)
app.include_router(cvat_admin_router)
app.include_router(health_router)
app.include_router(index_router)
app.include_router(jobs_router)
app.include_router(query_router)
app.include_router(realtime_router)
app.include_router(robot_router)
app.include_router(scene_router)
app.include_router(site_state_router)
app.include_router(webhook_router)


@app.get("/index/form", response_class=HTMLResponse, include_in_schema=False)
async def index_form():
    """Simple HTML form to upload a local video or submit a URL for indexing."""
    return get_index_form_html()
