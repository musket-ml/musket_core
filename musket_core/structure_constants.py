import os

FILE_NAME_ERROR_YAML = "error.yaml"

FILE_NAME_IN_PROGRESS_YAML = "inProgress.yaml"

FILE_NAME_CONFIG_CONCRETE_YAML = "config_concrete.yaml"

FILE_NAME_CONFIG_YAML = "config.yaml"

FILE_NAME_SUMMARY_YAML = "summary.yaml"

DIR_NAME_PREDICTIONS = "predictions"


def constructSummaryYamlPath(path:str):
    return os.path.join(path,FILE_NAME_SUMMARY_YAML)


def constructInProgressYamlPath(path:str):
    return os.path.join(path,FILE_NAME_IN_PROGRESS_YAML)


def constructConfigYamlPath(path:str):
    return os.path.join(path,FILE_NAME_CONFIG_YAML)


def constructConfigYamlConcretePath(path:str):
    return os.path.join(path,FILE_NAME_CONFIG_CONCRETE_YAML)


def constructErrorYamlPath(path:str):
    return os.path.join(path,FILE_NAME_ERROR_YAML)


def constructPredictionsDirPath(path:str):
    return os.path.join(path,DIR_NAME_PREDICTIONS)

def isNewExperementDir(dirPath:str):
    return os.path.exists(constructConfigYamlPath(dirPath)) and not os.path.exists(constructSummaryYamlPath(dirPath)) and not os.path.exists(constructInProgressYamlPath(dirPath))

def isExperimentDir(dirPath:str):
    return os.path.exists(constructConfigYamlPath(dirPath))