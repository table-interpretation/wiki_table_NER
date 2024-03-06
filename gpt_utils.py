# From gpt repo on code.siemens
# Configuration for GPT instances
import copy
import json
import os
import logging
from typing import Dict, List, Tuple, Union

import openai


# For authenticating with openai there are two ways of authentication:
# - An API key associated with an endpoint
# - a token for the service principal, which is retrieved using a secret associated with the service principal
# Try reading secrets from file.
# For secrets with API keys, contact Stefan Langer
# The secrets file contains API keys and/or service principal secrets
# Alternatively, some secrets can be retrieved from env variables


def read_endpoint_configuration(config_file: str) -> Dict:
    """
    Read the configuration file.

    :param config_file: The path to the files with the endpoint configurations
    :return:
    """
    with open(config_file, "r") as f:
        configuration_data = json.load(f)
    api_endpoints = configuration_data.get("api_endpoints", {})
    # Build endpoint - info-mapping for easier access
    configuration_data["endpoint_info"] = {}
    for api_endpoint in api_endpoints:
        endpoint = api_endpoint.get("endpoint", "")
        if endpoint:
            configuration_data["endpoint_info"][endpoint] = api_endpoint
    return configuration_data


configuration_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "configuration_files"
)
# Try to read the default secrets or the dummy secrets
endpoint_configuration = {}
file =  "../configuration_endpoints_smr_6.json"
secrets_file_path = file
if os.path.exists(secrets_file_path):
    logging.debug(f"Reading configuration from {secrets_file_path}")
    endpoint_configuration = read_endpoint_configuration(secrets_file_path)


# ID of the service principal (= functional user for API)
# ID secrets associated with the tenant are provided via:
# - secrets file (see above)
# - environment variables (secrets only)
# Service principal default tenant id for SMR
def get_sp_tenant_id() -> str:
    """
    Get the tenant id
    - from secrets
    - if not in secrets, return default
    :return: The tenant id (default or from secrets)
    """
    # Default value
    sp_tenant_id = "38ae3bcd-9579-4fd4-adda-b42e1495d55a"
    # Update from secrets file, if present
    if "service_principal" in endpoint_configuration:
        tmp = endpoint_configuration["service_principal"].get("tenant_id", "")
        if tmp:
            sp_tenant_id = tmp
        else:
            logging.warning(
                f"Could not retrieve service principal tenant id. Using default value"
            )
    return sp_tenant_id


def get_sp_secret() -> Tuple[str, str]:
    """
    Get the id and value for the service principal secret.
    Use the following order:
    - from secrets file
    - from env vars OPENAI_SECRET_ID, OPENAI_SECRET_VALUE
    :return: Tuple: Secret id, secret value. Empty strings if retrieval failed.
    """
    sp_secret_id = ""
    sp_secret_value = ""
    if "service_principal" in endpoint_configuration:
        sp_secret_id = endpoint_configuration["service_principal"].get("secret_id", "")
        sp_secret_value = endpoint_configuration["service_principal"].get(
            "secret_value", ""
        )
    if not sp_secret_id:
        id_env_var = ("OPENAI_SECRET_ID",)
        sp_secret_id = os.getenv(id_env_var, "")
        logging.error(
            f"Could not retrieve service principal secret id. Specify in secrets or with env variable {id_env_var}"
        )
    if not sp_secret_value:
        val_env_var = ("OPENAI_SECRET_VALUE",)
        sp_secret_value = os.getenv(val_env_var, "")
        logging.error(
            f"Could not retrieve service principal secret value. Specify in secrets or with env variable {val_env_var}"
        )

    return sp_secret_id, sp_secret_value


def get_api_key(gpt_config: Dict) -> str:
    """
    Get the API key for the given configuration.
    Try in following order:
    - "api_key" from the gpt_config
    - from secrets provided in secrets file (based on endpoint in gpt_config)
    - from env variable "OPENAI_API_KEY"
    :param gpt_config: The configuration to get the API key from.
    :return: The API key or an empty string
    """
    api_key = gpt_config.get("api_key", "")
    if not api_key:
        endpoint = gpt_config.get("endpoint")
        api_key = (
            endpoint_configuration["endpoint_info"].get(endpoint, {}).get("api_key", "")
        )
    if not api_key:
        api_key_env_var = "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env_var, "")
    if api_key:
        return api_key
    else:
        logging.error(
            f"Could not retrieve API key. Specify in secrets or with env variable {api_key_env_var}"
        )
        return ""


def get_api_version(gpt_config: Dict) -> str:
    """
    Get the API key for the given configuration.
    Try in following order:
    - "api_version" from the gpt_config
    - from secrets provided in secrets file (based on endpoint in gpt_config)
    :param gpt_config: The configuration to get the endpoint from.
    :return: The API version
    """
    api_version = gpt_config.get("api_version", "")
    if not api_version:
        endpoint = gpt_config.get("endpoint")
        api_version = (
            endpoint_configuration["endpoint_info"]
            .get(endpoint, {})
            .get("api_version", "")
        )
    return api_version


# Static information about the available models in openai
# Adapt if this changes
model_data = {
    "gpt-4": {"type": "chat", "max": 8191},
    "gpt-4-32k": {"type": "chat", "max": 32768},
    "gpt-4-turbo-128k-1106": {"type": "chat", "max": 64000},
    "gpt-35-turbo": {"type": "chat", "max": 4097},
    "gpt-35-turbo-16k": {"type": "chat", "max": 16385},
    "text-davinci-003": {"type": "completion", "max": 4097},
    "gpt-35-turbo-instruct": {"type": "completion", "max": 4097},
    "text-embedding-ada-002": {"type": "embedding", "max": 8191},
}


def get_model_type(model):
    """
    Get the type ('chat' or 'completion') of a given engine
    :param model: The model name (e.g. gpt-4-32k)
    :return: 'chat', 'completion' or '' if engine does not exist in data
    """
    return model_data.get(model, {"type": "", "max": 0})["type"]


def get_model_max_tokens(model):
    """
    Get the max number of tokens the engine supports
    :param model: the engine name (e.g. gpt-35-turbo)
    :return: The max number of tokens for the model or 0 if not in model data
    """
    return model_data.get(model, {"type": "", "max": 0})["max"]


# Our base configuration for a chat endpoint
gpt_config_base_chat = {
    "type": "chat",
    "max_prompt_tokens": 0,  # add, dependent on endpoint
    "endpoint": "",  # add endpoint
    "api_version": "",  # add api version
    "api_key": "",  # add api key
    "params": {  # These parameters are passed to the openai api endpoint
        "model": "",  # add model, e.g. gpt-4
        "temperature": 0,
        "max_tokens": 512,
        "top_p": 0.5,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    },
}

# Our base configuration for a completion endpoint
gpt_config_base_completion = {
    "type": "completion",
    "max_prompt_tokens": 0,  # add, dependent on endpoint
    "endpoint": "",  # add endpoint
    "api_version": "",  # add api version
    "api_key": "",  # add api key
    "params": {  # These parameters are passed to the openai api endpoint
        "model": "",  # add engine, e.g. text-davinci-003
        "temperature": 0,
        "max_tokens": 512,
        "top_p": 0.5,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "best_of": 1,
        "stop": None,
    },
}

# Our base configuration for an embedding endpoint
gpt_config_base_embedding = {
    "type": "embedding",
    "max_prompt_tokens": 0,  # add, dependent on endpoint
    "endpoint": "",  # add endpoint
    "api_version": "",  # add api version
    "api_key": "",  # add api key
    "params": {  # These parameters are passed to the openai api endpoint
        "model": "",  # add model, e.g. text-embedding-ada-002
    },
}


def build_gpt_configuration(endpoint: str, model: str) -> Union[Dict, None]:
    """
    For a given endpoint and engine, build the correct configuration
    :param endpoint: The endpoint to use - use full url
    :param model: The engine (e.g. gpt-4)
    :return: The full configuration or None, if we do not know the engine
    """
    model_type = get_model_type(model)
    # If this engine type if not known to us - return None
    if not model_type or model_type not in ["chat", "completion", "embedding"]:
        return None
    if model_type == "chat":
        conf = copy.deepcopy(gpt_config_base_chat)
    elif model_type == "completion":
        conf = copy.deepcopy(gpt_config_base_completion)
    elif model_type == "embedding":
        conf = copy.deepcopy(gpt_config_base_embedding)
    conf["endpoint"] = endpoint
    conf["api_key"] = get_api_key(conf)
    conf["api_version"] = get_api_version(conf)
    conf["max_prompt_tokens"] = get_model_max_tokens(model)
    conf["params"]["model"] = model
    return conf


def build_gpt_configurations(file_path="../configuration_endpoints_smr_6.json") -> List[Dict]:
    """
    Build the gpt configurations based on the endpoint configuration files.

    :return: A list of gpt  engine configurations.
    """
    if file_path:
        ep_configuration = read_endpoint_configuration(file_path)
    else:
        ep_configuration = endpoint_configuration
    gpt_configurations = []
    for endpoint_info in ep_configuration.get("api_endpoints", []):
        endpoint = endpoint_info["endpoint"]
        for model in endpoint_info["models"]:
            conf = build_gpt_configuration(endpoint, model)
            gpt_configurations.append(conf)
    return gpt_configurations


def get_gpt_configuration(endpoint: Union[str, None], model: str) -> Dict:
    """
    Build and select a specific configuration for an endpoint and a model.

    :param endpoint: The endpoint to get the configuration for. Can be None to use any endpoint
    :param model: The model to get the configration for
    :return:
    """
    configs = build_gpt_configurations()
    conf = None
    for gpt_conf in configs:
        if (not endpoint or gpt_conf["endpoint"] == endpoint) and gpt_conf["params"][
            "model"
        ] == model:
            conf = gpt_conf
    return conf


def read_gpt_config(endpoint, model_name):

    """Substitute with own endpoint configuration"""

    if model_name == "gpt-35-turbo-instruct":
        endpoint = "https://openai-aiattack-msa-000898-eastus-smrattack-00.openai.azure.com/"
    elif model_name in ["gpt-35-turbo", "gpt-35-turbo-16k", "text-embedding-ada-002"]:
        endpoint = "https://openai-aiattack-msa-000898-australiaeast-smrattack-00.openai.azure.com/"
    else:
        endpoint = "https://openai-aiattack-msa-000898-australiaeast-smrattack-00.openai.azure.com/"

    config = get_gpt_configuration(endpoint, model_name)

    openai.api_type = "azure"
    openai.api_base = config["endpoint"]
    openai.api_version = config["api_version"]
    openai.api_key = config["api_key"]
