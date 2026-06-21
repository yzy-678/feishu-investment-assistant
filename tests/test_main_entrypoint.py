from src import __main__


class TestMainEntrypoint:
    def test_main_uses_env_values(self, monkeypatch):
        called = {}

        def fake_run(app, host, port, reload):
            called["app"] = app
            called["host"] = host
            called["port"] = port
            called["reload"] = reload

        monkeypatch.setattr(__main__.uvicorn, "run", fake_run)
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("RELOAD", "true")

        __main__.main()

        assert called == {
            "app": "src.main:app",
            "host": "127.0.0.1",
            "port": 9000,
            "reload": True,
        }

    def test_main_uses_defaults(self, monkeypatch):
        called = {}

        def fake_run(app, host, port, reload):
            called["app"] = app
            called["host"] = host
            called["port"] = port
            called["reload"] = reload

        monkeypatch.setattr(__main__.uvicorn, "run", fake_run)
        monkeypatch.delenv("HOST", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("RELOAD", raising=False)

        __main__.main()

        assert called == {
            "app": "src.main:app",
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False,
        }
