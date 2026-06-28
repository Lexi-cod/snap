import os
import requests
from flask import Flask, render_template, request, Response, stream_with_context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

API_BASE = "http://localhost:8000"
TIMEOUT = 300


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    url = f"{API_BASE}/{path}"
    method = request.method

    try:
        if method == "GET":
            resp = requests.get(url, params=request.args, timeout=TIMEOUT, stream=True)
        elif method == "POST":
            if request.files:
                files = {
                    k: (v.filename, v.stream, v.content_type)
                    for k, v in request.files.items()
                }
                resp = requests.post(url, files=files, data=request.form.to_dict(),
                                     timeout=TIMEOUT, stream=True)
            else:
                resp = requests.post(
                    url,
                    json=request.get_json(silent=True),
                    headers={"Content-Type": "application/json"},
                    timeout=TIMEOUT,
                    stream=True,
                )
        elif method == "DELETE":
            resp = requests.delete(url, timeout=TIMEOUT, stream=True)
        else:
            resp = requests.request(method, url, timeout=TIMEOUT, stream=True)

        content_type = resp.headers.get("Content-Type", "application/json")

        if "text/event-stream" in content_type:
            def sse_stream():
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return Response(
                stream_with_context(sse_stream()),
                status=resp.status_code,
                content_type=content_type,
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        return Response(resp.content, status=resp.status_code, content_type=content_type)

    except requests.exceptions.ConnectionError:
        return Response(
            '{"error":"Cannot connect to API server on port 8000"}',
            status=502,
            content_type="application/json",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
