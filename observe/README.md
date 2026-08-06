# InferX monitoring stack

Start InferX on port 8000, then launch the core stack:

```bash
cd observe
GRAFANA_ADMIN_PASSWORD='choose-a-password' docker compose up -d
```

Add GPU telemetry when the host has the NVIDIA Container Toolkit and a
DCGM-compatible driver/GPU:

```bash
docker compose --profile gpu up -d
```

Add host CPU/memory/network telemetry with `--profile host`. Grafana is bound
to <http://127.0.0.1:3000> and Prometheus to
<http://127.0.0.1:9090>; neither UI is exposed publicly. The default Grafana
user is `admin`. Always override the development-only default password.

Prometheus reaches the host InferX endpoint through
`host.docker.internal:8000`. Change that target in `prometheus/prometheus.yml`
when InferX runs in another container or on another host. Use a private network
and authentication/TLS at ingress for any remote deployment.

For short benchmarks, the InferX scrape interval is five seconds. Set
`PROMETHEUS_RETENTION=24h` to reduce local storage. Diagnostic collective
timing is separately enabled on InferX, for example
`--collective-timing-sample-rate 128`; it is intentionally off by default.

Before starting the GPU profile, verify that the pinned DCGM Exporter image is
compatible with the rented host's NVIDIA driver. A provider-managed DCGM host
engine can be used instead if containers are not allowed `SYS_ADMIN`.
