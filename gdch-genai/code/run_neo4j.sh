#!bin/bash/

sudo docker run -it --rm \
        --publish=7474:7474 --publish=7687:7687 \
        --volume=$HOME/neo4j/data:/data \
        --volume=$HOME/neo4j/logs:/logs \
        --env NEO4J_PLUGINS='["graph-data-science"]' \
        neo4j:latest