# RTVis: Research Trend Visualization Toolkit

## Introduction

## Installation

### Method 1: Directly run the program (Support Windows, Linux, Mac OS)

1. Download the latest release by

```bash
git clone https://github.com/RTVis/RTVis.git
cd RTVis
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Preprocess the data

```bash
python dataprocessing.py
```

4. Run the program

```bash
python app.py
```

5. Open the browser and visit the website

```
http://localhost:8050/
```

### Method 2: Run the program in Docker (Support Windows, Linux, Mac OS)

1. Download the latest release by

```bash
git clone https://github.com/RTVis/RTVis.git
cd RTVis
```

2. Build the docker image

```bash
docker build -t rtvis .
```

3. Run the docker image

```bash
docker run -p 8050:8050 rtvis
```

4. Open the browser and visit the website

```
http://localhost:8050/
```

(Optional) 5. If you want to run the program in the background, you can use the following command

```bash
docker run -d -p 8050:8050 rtvis
```

Or you can use the `docker-compose.yml` file to run the program in the background

```bash
docker-compose up -d
```

