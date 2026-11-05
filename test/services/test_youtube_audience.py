"""Verify configuration, task snapshots and real HTTP form encoding of the
made-for-kids audience declaration, without contacting any external platform."""

from concurrent.futures import Future
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.config import config
from app.models.schema import VideoParams
from app.services import task
from app.services.state import MemoryState
from app.services.upload_post import UploadPostService


@pytest.fixture
def upload_config():
    """Tests use fictional credentials only and must never read the user's real
    publishing targets or trigger a real upload."""
    values = {
        "upload_post_enabled": True,
        "upload_post_api_key": "local-test-key",
        "upload_post_username": "local-test-user",
        "upload_post_platforms": ["youtube"],
    }
    with patch.object(config, "app", values):
        yield values


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize(
    "extra",
    [None, {}, {"selfDeclaredMadeForKids": False}, {"selfDeclaredMadeForKids": True}],
)
def test_audience_payload_and_snapshot_override(
    tmp_path, upload_config, configured, extra
):
    """An explicit task snapshot wins over current config; the declaration must
    also be sent when no metadata is provided."""
    upload_config["upload_post_youtube_made_for_kids"] = configured
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test-media")
    with patch("app.services.upload_post.requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "success": True,
            "results": {"youtube": {"success": True}},
        }
        result = UploadPostService().upload_video(
            str(video), "test", youtube_extra=extra
        )
    assert result["success"] is True
    data = post.call_args.kwargs["data"]
    expected = (extra or {}).get("selfDeclaredMadeForKids", configured)
    assert [v for k, v in data if k == "selfDeclaredMadeForKids"] == [
        str(expected).lower()
    ]


@pytest.mark.parametrize("invalid", ["false", "true", "", 0, 1, None, [], {}])
def test_invalid_audience_never_uploads(tmp_path, upload_config, invalid):
    """An invalid config must not be implicitly converted into an audience
    declaration, and the upload must not continue."""
    upload_config["upload_post_youtube_made_for_kids"] = invalid
    video = tmp_path / "video.mp4"
    video.touch()
    with patch("app.services.upload_post.requests.post") as post:
        result = UploadPostService().upload_video(str(video), "test")
    assert result["success"] is False
    assert "boolean" in result["error"]
    post.assert_not_called()


def test_other_platforms_ignore_youtube_audience(tmp_path, upload_config):
    """Non-YouTube publishing neither reads nor validates this setting, and a bad
    audience config must not affect it."""
    upload_config["upload_post_youtube_made_for_kids"] = "invalid"
    video = tmp_path / "video.mp4"
    video.touch()
    with patch("app.services.upload_post.requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "success": True,
            "results": {
                "tiktok": {"success": True},
                "instagram": {"success": True},
            },
        }
        result = UploadPostService().upload_video(
            str(video),
            "test",
            platforms=["tiktok", "instagram"],
            youtube_extra={"selfDeclaredMadeForKids": "invalid"},
        )
    assert result["success"] is True
    assert "selfDeclaredMadeForKids" not in dict(post.call_args.kwargs["data"])


@pytest.mark.parametrize("selected", [False, True])
def test_queued_audience_survives_config_change(upload_config, selected):
    """Defer the real worker function to verify that a config change while the
    item waits in the queue does not leak into an already submitted task."""
    upload_config["upload_post_youtube_made_for_kids"] = selected
    state = MemoryState()
    state.update_task("audience-snapshot", state=task.const.TASK_STATE_COMPLETE)
    future = Future()
    with (
        patch.object(task.sm, "state", state),
        patch.object(task, "_cross_post_slots", MagicMock()),
        patch.object(
            task._cross_post_executor, "submit", return_value=future
        ) as submit,
        patch.object(task.llm, "generate_social_metadata", return_value={}),
        patch.object(
            task.upload_post, "cross_post_video", return_value={"success": True}
        ) as upload,
    ):
        assert (
            task._schedule_cross_post(
                "audience-snapshot",
                ["one.mp4", "two.mp4"],
                VideoParams(video_subject="test"),
                "test",
                ["youtube"],
                "private",
                UploadPostService().youtube_made_for_kids,
            )
            is None
        )
        upload_config["upload_post_youtube_made_for_kids"] = not selected
        fn, *args = submit.call_args.args
        fn(*args)
        future.set_result(None)
    assert upload.call_count == 2
    assert all(
        c.kwargs["youtube_extra"]["selfDeclaredMadeForKids"] is selected
        for c in upload.call_args_list
    )


@pytest.mark.parametrize("selected", [None, False, True])
def test_real_http_multipart_audience(tmp_path, upload_config, selected):
    """Receive the multipart request over real local HTTP to verify the legacy
    configuration and the true/false encoding."""
    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
            )
            received.append(
                {
                    part.get_param(
                        "name", header="content-disposition"
                    ): part.get_payload(decode=True)
                    for part in message.iter_parts()
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true, "results": {"youtube": {"success": true}}}')

        def log_message(self, *_args):
            # The local receiver does not print request headers, so test output
            # never contains authentication data.
            pass

    if selected is not None:
        upload_config["upload_post_youtube_made_for_kids"] = selected
    video = tmp_path / "video.mp4"
    video.write_bytes(b"multipart-test-media")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        # Only loopback is contacted, bypassing any developer-machine proxy so
        # test requests are never handed to an external proxy service.
        with (
            requests.Session() as session,
            patch.object(
                UploadPostService, "API_BASE", f"http://127.0.0.1:{server.server_port}"
            ),
        ):
            session.trust_env = False
            with patch(
                "app.services.upload_post.requests.post", side_effect=session.post
            ):
                result = UploadPostService().upload_video(str(video), "test")
        assert result["success"] is True
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert received[0]["selfDeclaredMadeForKids"] == (b"true" if selected else b"false")
    assert received[0]["video"] == b"multipart-test-media"
