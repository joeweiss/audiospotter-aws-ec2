# Tests

## Ubuntu 20

Tests to come.

## M1 Mac using Docker

Note: Currently, the tests will utilize actual AWS resources (S3 buckets). These tests will be mocked in non-alpha releases.

```
docker-compose build
docker-compose up -d
docker-compose exec main pip install -r tests/docker-m1-test-requirements.txt
docker-compose exec main pytest --capture=no

```
