all:
	docker build --tag hello .

run:
	docker compose up -d

clean:
	docker compose down

test:
	docker compose exec app sh test.sh
	#docker build -f Dockerfile.test --tag test .
	#docker compose -f docker-compose-test.yml up -d
