import configparser
from neo4j import GraphDatabase

'''
# Aryan creds
neo4j_aura_uri = "neo4j+s://698e542b.databases.neo4j.io"
neo4j_aura_auth = ("neo4j", "VjFYhAF8bXPAd2l5XiLLsqlj6IIEIglBZuPkf_Th4jY")
'''

'''
# Akash Creds
neo4j_aura_uri = "neo4j+s://3e0f66cf.databases.neo4j.io"
neo4j_aura_auth = ("neo4j", "PAHm6sGmJV3E0IOFGvQKtI9U9pDlepcgcrMzEExyv7w")
'''

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config.read('config.ini')

neo4j_uri = config.get("CREDS", "neo4j_uri")
neo4j_user = config.get("CREDS", "neo4j_user")
neo4j_pass = config.get("CREDS", "neo4j_pass")

print(neo4j_uri)

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

def add_entity(entity, date):
    name = entity["name"]
    typ = entity["type"]
    if typ == "individual":
        driver.execute_query(
            "MERGE (a: entity {name: $name, type: $typ, date: $date})",
            name = name, typ = typ, date = date, database="neo4j",
        )
    else:
        driver.execute_query(
            "MERGE (a: entity {name: $name, type: $typ})",
            name = name, typ = typ, database="neo4j",
        )

    return

def entity_type(entity_name):
    results = driver.execute_query(
        "MATCH (a: entity {name: $entity_name})"
        "RETURN a",
        entity_name = entity_name, database="neo4j"
    )
    if len(results.records):
        return results.records[0].data()['a']['type']
    else:
        return 'NA'


def add_relation(edge, date):
    entity1 = edge["entity1"]
    relation = edge["relation"]
    entity2 = edge["entity2"]
    entity1_type = entity_type(entity1)
    entity2_type = entity_type(entity2)

    if entity1_type == 'NA' or entity2_type == 'NA':
        return

    driver.execute_query(
        f"""MERGE (a: entity {{name: $entity1{", date: $date" if entity1_type == "individual" else ""}}}) """
        f"""MERGE (b: entity {{name: $entity2{", date: $date" if entity2_type == "individual" else ""}}}) """
        f"MERGE (a) - [:{relation}] -> (b) ",
        entity1 = entity1, entity2=entity2, date = date, database="neo4j"
    )
    return

def add_property(ppt, date):
    entity = ppt["entity"]
    desc = ppt["description"]
    typ = entity_type(entity)
    if typ == 'NA':
        return
    
    driver.execute_query(
        f"""MERGE (a: entity {{name: $entity{", date: $date" if typ == "individual" else ""}}}) """
        "SET a.desc = coalesce(a.desc, []) + $desc",
        entity = entity, desc = desc, date = date, database="neo4j"
    )
    return


"""
if __name__ == "__main__":
    import os
    import json

    folder = "./logs/latest/"
    for file in os.listdir(folder):
        print(file)
        with open(folder+file, "r") as f:
            text = f.read()
        info = json.loads(text)["info"]


        entities = info["entities"]
        relations = info["relations"]
        properties = info["properties"]
        date = info["date"]

        for entity in entities:
            add_entity(entity, date)

        for edge in relations:
            add_relation(edge, date)

        for ppt in properties:
            add_property(ppt, date) 

    '''

    from pprint import pprint

    entities = driver.execute_query(
        "MATCH (a: entity {name: 'Farhan'})"
        "RETURN a",
    )

    print(entities['a'])

    for entity in entities:
        print(entity)
    '''

"""