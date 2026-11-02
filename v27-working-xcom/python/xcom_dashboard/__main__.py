"""Allows `py -m xcom_dashboard` as an alias for the uvicorn invocation."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("xcom_dashboard.app:app", host="127.0.0.1", port=8000, reload=False)
