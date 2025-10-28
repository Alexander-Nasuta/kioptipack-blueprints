import uuid

if __name__ == '__main__':
    res = uuid.uuid5(uuid.NAMESPACE_DNS, str({"a":1, "b":2}))
    print(f"UUID5: {res}")