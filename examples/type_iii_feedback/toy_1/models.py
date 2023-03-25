from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def cancer_bn(override_target=None):
    # Define the model structure
    model = BayesianNetwork(
        [
            ("Pollution", "Cancer"),
            ("Smoker", "Cancer"),
            ("Cancer", "Xray"),
            ("Potato", "Xray"),
            ("Cancer", "Dyspnoea"),
        ]
    )

    cpd_poll = TabularCPD(
        variable="Pollution",
        variable_card=2,
        values=[[0.9], [0.1]]
    )

    cpd_smoke = TabularCPD(
        variable="Smoker",
        variable_card=2,
        values=[[0.3], [0.7]]
    )

    cpd_potato = TabularCPD(
        variable="Potato",
        variable_card=2,
        values=[[0.4], [0.6]]
    )

    cpd_cancer = TabularCPD(
        variable="Cancer",
        variable_card=2,
        values=[
            [0.3, 0.95, 0.4, 0.4],
            [0.7, 0.05, 0.6, 0.6]
        ],
        evidence=["Smoker", "Pollution"],
        evidence_card=[2, 2],
    )
    cpd_xray = TabularCPD(
        variable="Xray",
        variable_card=2,
        values=[
            [0.4, 0.65, 0.4, 0.5],
            [0.6, 0.35, 0.6, 0.5]
        ],
        evidence=["Cancer", "Potato"],
        evidence_card=[2, 2],
    )

    cpd_dysp = TabularCPD(
        variable="Dyspnoea",
        variable_card=2,
        values=[
            [0.65, 0.3],
            [0.35, 0.7]
        ],
        evidence=["Cancer"],
        evidence_card=[2],
    )

    # Associating the parameters with the model structure.
    model.add_cpds(cpd_poll, cpd_smoke, cpd_potato, cpd_cancer, cpd_xray, cpd_dysp)

    # Check if the model is valid
    model.check_model()

    return model, "Xray" if not override_target else override_target

def toy_bn(override_target=None):
    # Define the network structure
    model = BayesianNetwork(
        [
            ('X5', 'Y'),
            ('X6', 'Y'),
            ('X7', 'Y'),
            ('Y', 'X1'),
            ('Y', 'X2'),
            ('Y', 'X3'),
            ('X4', 'X3'),
            ('X8', 'X4'),
        ]
    )

    # Define CPDs with arbitrary probability values
    cpd_X5 = TabularCPD(variable='X5', variable_card=2, values=[[0.6], [0.4]])
    cpd_X6 = TabularCPD(variable='X6', variable_card=2, values=[[0.7], [0.3]])
    cpd_X7 = TabularCPD(variable='X7', variable_card=2, values=[[0.8], [0.2]])
    cpd_X8 = TabularCPD(variable='X8', variable_card=2, values=[[0.9], [0.1]])

    cpd_Y = TabularCPD(
        variable='Y',
        variable_card=2,
        values=[
            [0.3, 0.4, 0.7, 0.6, 0.1, 0.8, 0.3, 0.4],
            [0.7, 0.6, 0.3, 0.4, 0.9, 0.2, 0.7, 0.6],
        ],
        evidence=['X5', 'X6', 'X7'],
        evidence_card=[2, 2, 2],
    )

    cpd_X1 = TabularCPD(
        variable='X1',
        variable_card=2,
        values=[
            [0.9, 0.1],
            [0.1, 0.9]
        ],
        evidence=['Y'],
        evidence_card=[2]
    )
    cpd_X2 = TabularCPD(
        variable='X2',
        variable_card=2,
        values=[
            [0.8, 0.6],
            [0.2, 0.4]
        ],
        evidence=['Y'],
        evidence_card=[2]
    )
    cpd_X3 = TabularCPD(
        variable='X3',
        variable_card=2,
        values=[
            [0.7, 0.3, 0.5, 0.5],
            [0.3, 0.7, 0.5, 0.5]
        ],
        evidence=['Y', 'X4'],
        evidence_card=[2, 2]
    )
    cpd_X4 = TabularCPD(
        variable='X4',
        variable_card=2,
        values=[
            [0.6, 0.4],
            [0.4, 0.6]
        ],
        evidence=['X8'],
        evidence_card=[2]
    )

    # Add CPDs to the model
    model.add_cpds(cpd_X5, cpd_X6, cpd_X7, cpd_Y, cpd_X1, cpd_X2, cpd_X3, cpd_X4, cpd_X8)

    # Check if the model is consistent
    assert model.check_model(), "The model is inconsistent"

    return model, "Y" if not override_target else override_target


if __name__ == "__main__":
    toy_bn()
