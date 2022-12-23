FROM python:3.9-bullseye

RUN pip install poetry==1.1.15

WORKDIR /app

COPY poetry.lock pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install #--no-interaction #--no-ansi
RUN poetry run prefect config set PREFECT_API_URL=http://orion_server:4200/api#http://host.docker.internal:4200/api

COPY . .

CMD ["./wait-for-it.sh", "orion_server:4200", "--", "poetry","run", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
