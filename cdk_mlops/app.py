#!/usr/bin/env python3

import aws_cdk as cdk

from cdk_mlops.cdk_mlops_stack import CdkMlopsStack


app = cdk.App()
CdkMlopsStack(app, "cdk-mlops")

app.synth()
