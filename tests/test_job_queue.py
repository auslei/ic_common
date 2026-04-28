import asyncio
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from src.common.job_queue import JobQueue, JobRecord

@pytest.fixture
def job_queue():
    return JobQueue("test_service")

@pytest.mark.asyncio
async def test_enqueue_basic(job_queue):
    job = job_queue.enqueue("test_job", {"data": 123})
    assert job.job_id is not None
    assert job.job_type == "test_job"
    assert job.status == "queued"
    assert job_queue.queue_size == 1
    assert job_queue.get_job_position(job.job_id) == 1

@pytest.mark.asyncio
async def test_deduplication(job_queue):
    job1 = job_queue.enqueue("test_job", {"data": 123})
    job2 = job_queue.enqueue("test_job", {"data": 123})
    assert job1.job_id == job2.job_id
    assert job_queue.queue_size == 1

@pytest.mark.asyncio
async def test_worker_execution(job_queue):
    async def run_fn(job):
        return {"result": job.request["val"] * 2}
    
    await job_queue.start(run_fn)
    job = job_queue.enqueue("calc", {"val": 10})
    
    # Wait for completion
    for _ in range(20):
        if job.status == "completed":
            break
        await asyncio.sleep(0.05)
        
    assert job.status == "completed"
    assert job.result == {"result": 20}
    await job_queue.stop()

@pytest.mark.asyncio
async def test_job_cancellation_queued(job_queue):
    job = job_queue.enqueue("test", {})
    assert job_queue.cancel_job(job.job_id) is True
    assert job.status == "failed"
    assert job.error == "Cancelled by user"

@pytest.mark.asyncio
async def test_job_cancellation_running(job_queue):
    start_event = asyncio.Event()
    async def run_fn(job):
        start_event.set()
        await asyncio.sleep(10) # Long job
        return "done"

    await job_queue.start(run_fn)
    job = job_queue.enqueue("test", {})
    
    await start_event.wait()
    assert job.status == "running"
    
    assert job_queue.cancel_job(job.job_id) is True
    
    # Wait for cleanup
    for _ in range(20):
        if job.status == "failed":
            break
        await asyncio.sleep(0.05)
        
    assert job.status == "failed"
    assert job.error == "Cancelled by user"
    await job_queue.stop()

@pytest.mark.asyncio
async def test_concurrency(job_queue):
    job_queue.max_concurrency = 2
    job_queue._semaphore = asyncio.Semaphore(2)
    
    running_count = 0
    max_observed_running = 0
    
    async def run_fn(job):
        nonlocal running_count, max_observed_running
        running_count += 1
        max_observed_running = max(max_observed_running, running_count)
        await asyncio.sleep(0.1)
        running_count -= 1
        return "ok"

    await job_queue.start(run_fn)
    job_queue.enqueue("j1", {})
    job_queue.enqueue("j2", {})
    job_queue.enqueue("j3", {})
    
    # Wait for all to finish
    for _ in range(50):
        counts = job_queue.job_counts()
        if counts["completed"] == 3:
            break
        await asyncio.sleep(0.05)
        
    assert max_observed_running == 2
    await job_queue.stop()

@pytest.mark.asyncio
async def test_trimming(job_queue):
    with patch("src.common.job_queue.MAX_COMPLETED_JOBS", 2):
        async def run_fn(job): return "ok"
        await job_queue.start(run_fn)
        
        j1 = job_queue.enqueue("j1", {"i": 1})
        j2 = job_queue.enqueue("j2", {"i": 2})
        j3 = job_queue.enqueue("j3", {"i": 3})
        
        for _ in range(50):
            if job_queue.job_counts()["completed"] == 3:
                break
            await asyncio.sleep(0.05)
            
        # j1 should have been trimmed (MAX=2, we have j1,j2,j3)
        assert job_queue.get_job(j1.job_id) is None
        assert job_queue.get_job(j2.job_id) is not None
        assert job_queue.get_job(j3.job_id) is not None
        await job_queue.stop()

from unittest.mock import patch

@pytest.mark.asyncio
async def test_router(job_queue):
    app = FastAPI()
    app.include_router(job_queue.make_router())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Enqueue via internal API
        job = job_queue.enqueue("api_test", {"x": 1})
        
        # List jobs
        resp = await ac.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        
        # Get job
        resp = await ac.get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job.job_id
        
        # Cancel job
        resp = await ac.delete(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        assert job.status == "failed"
        
        # Not found
        resp = await ac.get("/jobs/missing")
        assert resp.status_code == 404
