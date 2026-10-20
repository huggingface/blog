---
title: "Building CI Observability for Transformers with pytest, OpenTelemetry, and Grafana"
thumbnail: /blog/assets/96_hf_bitsandbytes_integration/Thumbnail_blue.png
authors:
- user: tarekziade
- user: ydshieh
---

# Building CI Observability for Transformers with pytest, OpenTelemetry, and Grafana


## The CircleCI Era

XXX explain what we did with circle CI
historicaly only CPU , we added GPU in our own github runners


## Moving all runners to our infrastructure

XXX explain how we moved all runners to our infra

## The Gap : Observability

XXX GH Action alone dont give us the Observability we need.
XXX we needed a circle CI replacement and build our custom UI


## Pytest and OpenTelemetry

XXX explain how it was simple to instrument our pytest suite with OpenTelemetry

XXX show the plugin and a small example

## Grafana, Tempo and Prometheus

XXX Describe the Grafana and Tempo stack
XXX describe the plumbing between grafana and github events to make grafana as live as possible

## Grafana as a wrapper

XXX Explain that more and more poanels are served by a custom Py web server
XXX in iframes. Grafana is just becoming a a grid
XXX an the py app is more efficient using tempo as its DB

## What we plan next


XXX

