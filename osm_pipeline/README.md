# OSM Pipeline and OSRM Docker Setup

This pipeline builds FrodoBots navigation data by selecting trajectory segments
that align with OpenStreetMap pedestrian routes and rendering frame-level
ego-centric OSM map images.

The routing backend is local OSRM through Docker. No Dockerfile is required: the
scripts use the official `osrm/osrm-backend` image.

## What Is Tracked

Tracked in Git:

- `scripts/preprocess_osrm.sh`: converts `.osm.pbf` files into OSRM MLD data.
- `scripts/start_osrm.sh`: starts/stops local OSRM Docker servers.
- `scripts/run_pipeline.sh`: runs episode selection and OSM map generation.
- `py/episode_selector.py`: selects good trajectory segments.
- `py/osm_map_generator.py`: renders per-frame map images.

Not tracked in Git:

- `osrm/`: raw `.osm.pbf` files and generated `.osrm*` files.
- `tile_cache/`: downloaded OSM map tiles.
- `osm_data/output_*/osm_maps*/`: generated map images.

These directories can be tens of gigabytes and are intentionally ignored.

## Directory Layout

Expected local layout:

```text
osm_pipeline/
├── osrm/
│   ├── hubei-latest.osm.pbf
│   ├── philippines-latest.osm.pbf
│   ├── italy-latest.osm.pbf
│   ├── new-zealand-latest.osm.pbf
│   ├── florida-latest.osm.pbf
│   ├── great-britain-latest.osm.pbf
│   ├── spain-latest.osm.pbf
│   ├── wuhan/
│   ├── manila/
│   ├── rome/
│   ├── wellington/
│   ├── florida/
│   ├── brighton/
│   └── madrid/
├── scripts/
└── py/
```

You can also keep OSRM data outside the repo and pass `OSRM_DIR`:

```bash
OSRM_DIR=/path/to/osrm_data bash osm_pipeline/scripts/start_osrm.sh rides11
```

## OSRM Regions and Ports

The routing scripts and Python pipeline use these local ports:

| Region | Port | Dataset group |
| --- | ---: | --- |
| Perth | 5001 | rides_00 legacy |
| Taipei | 5002 | rides_00 legacy |
| Tokyo | 5003 | rides_00 legacy |
| Wuhan | 5004 | output_rides_11 |
| Manila | 5005 | output_rides_11 |
| Rome | 5006 | output_rides_11 |
| Wellington | 5007 | output_rides_11 |
| Florida | 5008 | output_rides_11 |
| Brighton | 5009 | output_rides_11 |
| Madrid | 5010 | output_rides_11 |

The same mapping appears in `py/episode_selector.py` and
`py/osm_map_generator.py`.

## Prepare OSRM Data

Install Docker and place the required `.osm.pbf` files in `osm_pipeline/osrm/`.
The rides_11 preprocessing script expects:

| Region | Required PBF |
| --- | --- |
| Wuhan | `hubei-latest.osm.pbf` |
| Manila | `philippines-latest.osm.pbf` |
| Rome | `italy-latest.osm.pbf` |
| Wellington | `new-zealand-latest.osm.pbf` |
| Florida | `florida-latest.osm.pbf` |
| Brighton | `great-britain-latest.osm.pbf` |
| Madrid | `spain-latest.osm.pbf` |

Check what is present:

```bash
cd osm_pipeline
bash scripts/preprocess_osrm.sh status
```

Preprocess one region:

```bash
bash scripts/preprocess_osrm.sh wuhan
```

Preprocess all rides_11 regions:

```bash
bash scripts/preprocess_osrm.sh rides11
```

Internally, each region runs:

```bash
docker run --rm -v "$out_dir:/data" osrm/osrm-backend \
  osrm-extract -p /usr/local/share/osrm/profiles/foot.lua /data/<region>.osm.pbf

docker run --rm -v "$out_dir:/data" osrm/osrm-backend \
  osrm-partition /data/<region>.osrm

docker run --rm -v "$out_dir:/data" osrm/osrm-backend \
  osrm-customize /data/<region>.osrm
```

## Start OSRM Servers

Start all rides_11 OSRM servers:

```bash
cd osm_pipeline
bash scripts/start_osrm.sh rides11
```

Start every configured region:

```bash
bash scripts/start_osrm.sh all
```

Check status:

```bash
bash scripts/start_osrm.sh status
```

Stop all containers:

```bash
bash scripts/start_osrm.sh stop
```

Each server uses:

```bash
docker run -d --rm \
  --name osrm_<region> \
  -p <host_port>:5000 \
  -v "<region_osrm_dir>:/data" \
  osrm/osrm-backend \
  osrm-routed --algorithm mld /data/<region>.osrm
```

## Run the Pipeline

The Python pipeline expects the `mbra` conda environment and local OSRM servers.

```bash
cd osm_pipeline
bash scripts/run_pipeline.sh
```

Run map generation for a specific episode/segment:

```bash
bash scripts/run_pipeline.sh --skip-selection --ep 405 --seg 2
```

The pipeline stages are:

1. `py/episode_selector.py`: split moving segments and filter them by OSRM
   pedestrian-route alignment.
2. `py/osm_map_generator.py`: download/cache OSM tiles and render ego-centric
   `224x224` map images.

Generated outputs should remain local and ignored by Git.
