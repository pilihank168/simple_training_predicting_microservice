all:
	docker build -f Dockerfile --tag hello .
	docker build -f Dockerfile.pref --tag pref .

run:
	docker compose up -d

clean:
	docker compose down

test:
	docker compose exec app ./wait-for-it.sh orion_server:4200 -- sh test.sh
