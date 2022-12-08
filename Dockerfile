FROM python:3.9-bullseye

RUN pip install poetry==1.1.15

WORKDIR /app

COPY poetry.lock pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install #--no-interaction #--no-ansi

COPY . .

CMD ["poetry","run", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
