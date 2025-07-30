import yaml, json

def parse_json(json_str):
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].rsplit("```", 1)[0].strip()
        return json.loads(json_str)
    else:
        return json.loads(json_str.strip())

def parse_yaml(yaml_str):
    if "```yaml" in yaml_str:
        yaml_str = yaml_str.split("```yaml")[1].rsplit("```", 1)[0].strip()
        return yaml.safe_load(yaml_str)
    else:
        return yaml.safe_load(yaml_str.strip())

# Custom Dumper to enforce double quotes for strings
class DoubleQuoteDumper(yaml.Dumper):
    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')
DoubleQuoteDumper.add_representer(str, DoubleQuoteDumper.represent_str)

def json_to_yaml_str(json_data):
    # Dump it into a YAML string with the custom Dumper
    yaml_string = yaml.dump(json_data, Dumper=DoubleQuoteDumper, default_flow_style=False, sort_keys=False)
    return yaml_string 