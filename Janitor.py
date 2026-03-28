import numpy as np
from modelNN import predict_all, get_blame
from Test import TestClean

def define_mapping_matrix(totalfeatures, originalfeatures, transforms):
    mapping = np.zeros((originalfeatures, totalfeatures))

    singleparent = originalfeatures * transforms

    for i in range(singleparent):
        parent = i // transforms
        mapping[parent, i] = 1.0

    count = singleparent
    for i in range(singleparent):
        parent_a = i // 5
        for j in range(i+1, singleparent):
            parent_b = j // 5

            if parent_a == parent_b:
                # Interaction of the same chemical (e.g., Alcohol * Alcohol^2)
                mapping[parent_a, count] = 1.0
            else:
                # Interaction of two different chemicals
                mapping[parent_a, count] = 0.5
                mapping[parent_b, count] = 0.5
            count += 1
    return mapping
## hard coded feature numbers here
mapping = define_mapping_matrix(1540, 11, 5)



def Clean(janMat0: np.ndarray, janMat1, mW0, mW1, mB0, mB1, Inputs: np.ndarray, Quality: np.ndarray, matBin, Base_LR,epoch: int) -> np.ndarray:
    counts = np.bincount(matBin.flatten(), minlength=13)
    member_weights = np.where(counts > 0, 1.0 / counts, 0)
    equalizer = member_weights[matBin.flatten()].reshape(-1, 1)

    trainError, trainSigL1 = predict_all(Inputs, Quality, mW0, mW1, mB0, mB1, matBin)
    trainGrad0 = Inputs.T @ get_blame(mW1, trainSigL1, trainError * equalizer)
    trainGrad1 = trainSigL1.T @ trainError / Inputs.shape[0]
    quizGrad0, quizGrad1, quizSigL1 = TestClean(mW0, mW1, mB0, mB1)

    print(f"trainGrad0 shape: {trainGrad0.shape}")
    print(f"trainGrad1 shape: {trainGrad1.shape}")
    print(f"quizGrad0 shape: {quizGrad0.shape}")
    print(f"quizGrad1 shape: {quizGrad1.shape}")
    print(f"quizSigL1 shape: {quizSigL1.shape}")
    print(f"Inputs shape: {Inputs.shape}")

    janMat0 = CleanW0(janMat0, mW0, trainGrad0, quizGrad0, epoch)
    mW0, mW1, janMat0 = CleanW1(janMat0, mW0, mW1, trainGrad1, quizGrad1, quizSigL1, trainSigL1, epoch)
    Base_LR = UpdateB_LR(Base_LR)

    return janMat0, Base_LR, mW0, mW1
def CleanW0(janMat0: np.ndarray, mW0, trainGrad0, quizGrad0, epoch: int) -> np.ndarray:

    traitor_mask = (np.sign(trainGrad0) != np.sign(quizGrad0))
    janMat0[traitor_mask] = 0

    threshold = .05
    unstable = (np.abs(trainGrad0 - quizGrad0) > (threshold * np.abs(mW0)))
    invalids = unstable

    score = abs(trainGrad0)

    if epoch > 2000:
        score = abs(trainGrad0 - quizGrad0)

    print(f"train gradient shape: {trainGrad0.shape}")

    percentActive: int = int(round(np.sum(janMat0) * .2))

    for i in range(mW0.shape[1]):
        current_score = np.abs(trainGrad0[:, i])
        current_candidates = (janMat0[:, i] == 1) & (invalids[:, i] == True)
        #candidates = (janMat0 == 1) & (invalids == True)
        candidatesI = np.where(current_candidates)[0]

        if len(candidatesI) > 0:
            candidateScores = current_score[candidatesI]
            rankedIndex = np.argsort(candidateScores)[::-1]
            to_fire_indices = candidatesI[rankedIndex[:percentActive]]

            janMat0.ravel()[to_fire_indices] = 0

    print(np.sum(janMat0 == 0))

    return janMat0

def CleanW1(janMat0: np.ndarray, mW0, mW1, trainGrad1, quizGrad1, quizSigL1, trainSigL1, epoch: int) -> np.ndarray:

    demerits = np.zeros((mW1.shape[0], 1))

    simW0 = mapping @ np.abs(mW0)
    simW0 /= (np.sum(simW0, axis = 0) + 1e-9)
    simW0 = np.abs(np.corrcoef(simW0.T))

    simW1 = np.abs(np.corrcoef(trainSigL1.T))

    similarity = (0.5 * simW0) + (0.5 * simW1)

    totalSim = np.max(similarity - np.eye(mW1.shape[0]), axis=0)
    simDemerits = (totalSim // 0.15) * 2
    print(f"demerits shape: {demerits.shape}")
    print(f"simdemerits shape: {simDemerits.shape}")
    demerits += simDemerits.reshape(-1,1)

    traitor = (np.sign(trainGrad1.ravel()) != np.sign(quizGrad1.ravel()))
    demerits[traitor] += 8

    threshold = 0.05
    unstable = (np.abs(trainGrad1.ravel() - quizGrad1.ravel()) > (threshold * np.abs(mW1.ravel())))
    demerits[unstable] += 4

    w_mags = np.abs(mW1.ravel())
    demerits[w_mags < 0.50] += 1
    demerits[w_mags < 0.3] += 2
    demerits[w_mags < 0.1] += 2

    worst_to_best = np.argsort(demerits.flatten())[::-1]

    # Loop through only the top 3 offenders
    for i in worst_to_best[:15]:
        score = demerits[i]

        if score >= 1:
            if score % 2 == 0:
                mW0[:, i] *= (1 - (min(score, 5) / 10))
            else:
                mW0[:, i] *= (1 + (min(score, 5) / 10))
            print(f"💀 Neuron {i}: RESET (Score {score})")

        if score >= 5:
            magnitudes = np.abs(mW0[:, i])
            topindices = np.argsort(magnitudes)[-10:]
            janMat0[topindices, i] = 0

        if score >= 10:
            percentage = min(((score - 5) * .1), 1)

            num_to_mutate = int(percentage * mW0.shape[0])

            mutation_indices = np.random.choice(mW0.shape[0], num_to_mutate, replace=False)
            mW0[mutation_indices, i] = np.random.uniform(-0.1, 0.1, num_to_mutate)
            janMat0[mutation_indices, i] = 1

    return mW0, mW1, janMat0


def UpdateB_LR(Base_LR: float) -> float:
    return max((Base_LR * .675), 0.01)