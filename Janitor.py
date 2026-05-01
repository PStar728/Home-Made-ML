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

    # 1. Are they flatlined?
    print(f"Activation Volatility: {np.std(trainSigL1, axis=0).mean():.6f}")

    # 2. Are they mirrors or clones? (Check without np.abs)
    raw_corr = np.corrcoef(trainSigL1.T)
    print(f"Raw Correlation Range: {np.min(raw_corr):.2f} to {np.max(raw_corr):.2f}")

    # 3. What is the most common feature?
    top_features = [np.argmax(np.abs(mW0[:, i])) for i in range(16)]
    print(f"Top Feature indices: {top_features}")

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
    # 1. Recovery: Increment trust for connections on probation
    janMask_probation = ((janMat0 > 0) & (janMat0 < 1))
    janMat0[janMask_probation] = np.minimum(janMat0[janMask_probation] + 0.1, 1.0)

    # 2. Traitor Flip
    traitor = (np.sign(trainGrad1.ravel()) != np.sign(quizGrad1.ravel()))
    mW1.ravel()[traitor] *= -1.0

    num_neurons = mW1.shape[0]

    # 3. THE MUTUAL AUDIT LOOP
    for i in range(num_neurons):
        # simScores is a list of 15 similarities (Target i vs all others)
        simScores = GetSimilarities(i, mW0, mW1, trainSigL1, mapping)

        # We need the REAL indices of the 15 rivals (0-15 excluding i)
        rival_indices = np.arange(num_neurons)[np.arange(num_neurons) != i]

        # Audit each of the 15 rivals
        for idx_in_15, rival_id in enumerate(rival_indices):
            current_sim = simScores[idx_in_15]

            # 4. Apply threshold logic to this specific rival
            if current_sim > 0.40:
                # Calculate points for this specific similarity
                points = int(((current_sim - 0.25) // 0.15) * 2)

                if points > 0:
                    # Find the strongest features for the MAIN neuron
                    # (We use the Main's features because that's what the rival is stealing)
                    main_weights = np.abs(mW0[:, i])
                    num_to_kill = min(points, 10)
                    toCripple = np.argsort(main_weights)[-num_to_kill:]

                    # PUNISH THE RIVAL: Cripple its ability to use those features
                    # Target row: toCripple, Target column: rival_id
                    janMat0[toCripple, rival_id] = 0.1
                    mW0[toCripple, rival_id] *= 0.1

                    print(
                        f"⚖️ Neuron {i} caught Neuron {rival_id} copying ({current_sim:.2f}). Crippling {num_to_kill} features for {rival_id}.")

    return mW0, mW1, janMat0


def GetSimilarities(mainindex, mW0, mW1, matSig1, mapping):

    #simW0 = mapping @ np.abs(mW0)
    #simW0 /= (np.sum(simW0, axis=0) + 1e-9)
    simW0 = mW0
    # 1. Grab the weights for the main neuron
    mainW0 = simW0[:, mainindex]  # Shape: (1540,)
    mainMagW0 = np.sum(mainW0 ** 2) ** 0.5  # Result: Scalar

    # 2. Grab weights for all other 15 neurons
    comparedW0 = simW0[:, np.arange(simW0.shape[1]) != mainindex]  # Shape: (1540, 15)

    # FIX A: Use axis=0 to get 15 different magnitudes instead of one total sum
    comparedMagW0 = np.sum(comparedW0 ** 2, axis=0) ** 0.5  # Result: Array of 15

    # FIX B: Use newaxis to align the (1540,) vector with the (1540, 15) matrix
    # and use axis=0 to get 15 individual dot products
    dot_W0 = np.sum(mainW0[:, np.newaxis] * comparedW0, axis=0)  # Result: Array of 15

    # 3. Final division (NumPy does this element-wise)
    simDNA = dot_W0 / (mainMagW0 * comparedMagW0)  # Result: Array of 15



    mainSig = matSig1[:, mainindex]
    mainMagSig = np.sum(mainSig ** 2) ** 0.5

    comparedSig = matSig1[:, np.arange(matSig1.shape[1]) != mainindex]
    comparedMagSig = np.sum(comparedSig ** 2, axis=0) ** 0.5

    dot_Sig = np.sum(mainSig[:, np.newaxis] * comparedSig, axis=0)

    simSig = dot_Sig / (mainMagSig * comparedMagSig)

    totalSim = (0.65 * simDNA) + (0.35 * simSig)

    return totalSim


    simW0 = mapping @ np.abs(mW0)
    simW0 /= (np.sum(simW0, axis=0) + 1e-9)
    simTarget = simW0[:, mainindex]

    dot = simW0.T @ simTarget
    norm = np.linalg.norm(simW0, axis = 0) * np.linalg.norm(simTarget)
    simDNA = np.abs(dot / norm + 1e-9)

    simW1 = np.abs(np.corrcoef(mW1.T))
    simSig = np.abs(np.corrcoef(matSig1.T))

    similarity = (0.5 * simDNA) + (0.25 * simSig) + (0.25 * simW1)

    final = np.delete(similarity, mainindex)

    return final




def UpdateB_LR(Base_LR: float) -> float:
    return max((Base_LR * .675), 0.01)