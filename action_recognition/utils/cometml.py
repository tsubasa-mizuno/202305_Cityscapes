from comet_ml import Experiment


def comet():
    experiment = Experiment(
        api_key="nqdXsFWZqh06eZy2P1ZRE88RD",
        project_name="ActionRecognition",
        workspace="taikisugiura",
        auto_output_logging="simple",
    )
    return experiment
