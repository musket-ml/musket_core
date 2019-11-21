#### Building Musket ML image

Building musket_ml image

```
FROM python:3.6-buster

WORKDIR /usr/src/app
RUN pip install musket_ml
```

```
cd musket_ml
docker image build -t pythonmusket:1.0 .
```

#### build and run server image
```
cd ../server
docker image build -t server:1.0 .
docker container run --publish 8000:9393 --detach --name server_instance server:1.0
```

where 8000 - host port, 9393 docker instance port
#### redirect docker instance's stderr and stdout to current terminal
```
docker container attach server_instance
```

#### stop instance
```
docker container rm --force server_instance
```