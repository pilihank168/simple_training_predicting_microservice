all:
	docker build --tag hello .

run:
	docker compose up -d

clean:
	docker compose down

test:
	docker compose exec app sh test.sh
