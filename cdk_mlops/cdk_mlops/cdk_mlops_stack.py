from constructs import Construct
from aws_cdk import (
    Duration,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    cloudformation_include as cfn_inc,
)


class CdkMlopsStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # establishing parameters 
        s3_output_directory = 'S3_LOCATION' # TO BE FILLED BY USER
        s3_code_directory = 'S3_LOCATION' # TO BE FILLED BY USER
        s3_input_directory = 'S3_LOCATION' # TO BE FILLED BY USER
        step1_name = 'Data preprocessing'
        step2_name = 'Feature engineering (training)'
        step3_name = 'Feature engineering (validation)'
        step4_name = 'Model training'
        emr_cluster_name = 'CDKCreatedSteps'
        master_instance_type = 'm5.xlarge'
        core_instance_type = master_instance_type
        no_core_instances = 1
        model_type = 'gbt'
        model_params = "{'maxIter': 32}"
        subnet_id = 'YOUR_SUBNET' # TO BE FILLED BY USER
        params = {
                'MasterInstanceType': master_instance_type,
                'CoreInstanceType': core_instance_type,
                'NumberOfCoreInstances': no_core_instances,
                'EMRClusterName': emr_cluster_name,
                'SubnetID': subnet_id,
                'LogUri':f'{s3_output_directory}/logs',
                'BootstrapURL': f'{s3_code_directory}/bootstrap.sh',

                'Step1Name': step1_name,
                'Step1InputURLShippingLogs': f'{s3_input_directory}/ShippingLogs.csv',
                'Step1InputURLProductDescriptions':f'{s3_input_directory}/ProductDescriptions.csv',
                'Step1OutputURL':f'{s3_output_directory}/data_preprocessing',
                'Step1CodeFileURL':f'{s3_code_directory}/data_preprocessing.py',

                'Step2Name': step2_name,
                'Step2CodeFileURL':f'{s3_code_directory}/feature_engineering.py',
                'Step2InputURL': f'{s3_output_directory}/data_preprocessing/training',
                'Step2OutputURL': f'{s3_output_directory}/feature_engineering/training',

                'Step3Name': step3_name,
                'Step3CodeFileURL':f'{s3_code_directory}/feature_engineering.py',
                'Step3InputURL':f'{s3_output_directory}/data_preprocessing/validation',
                'Step3OutputURL':f'{s3_output_directory}/feature_engineering/validation',
                'Step3OutputURLModelDir': f'{s3_output_directory}/feature_engineering/training/pipeline_model',

                'Step4Name': step4_name,
                'Step4CodeFileURL': f'{s3_code_directory}/model_training.py',
                'Step4InputURLTraining': f'{s3_output_directory}/feature_engineering/training/dataframe',
                'Step4InputURLValidation': f'{s3_output_directory}/feature_engineering/validation/dataframe',
                'Step4OutputURL': f'{s3_output_directory}/model_training',
                'Step4ModelType': model_type,
                'Step4ModelParams': model_params
                }
        # creating resources
        template = cfn_inc.CfnInclude(self, "Template",  
            template_file="Multistep_cluster_CFN_4steps.json",
            parameters=params)
