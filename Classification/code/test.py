def addROC(actual, scores, color, label):
    actual = np.array(actual.copy())
    scores = np.array(scores.copy())

    # sortedscores = sorted(scores, key=itemgetter(1))
    scores = list(scores)
    DepictROCCurve(actual, scores, label, color, fname)


def GetRates(actual, scores):
    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractual = len(actual)
    nrdecoys = len(scores) - len(actual)

    foundactual = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actual:
            foundactual += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactual / float(nractual))
        fpr.append(founddecoys / float(nrdecoys))

    return tpr, fpr


def SetupROCCurvePlot(plt):
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)


def SaveROCCurvePlot(plt, randomline=True):
    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)


def AddROCCurve(plt, actual, scores, label, color):
    tpr, fpr = GetRates(actual, scores)

    plt.plot(fpr, tpr, color=color, linewidth=2, label=label)


def DepictROCCurve(actual, scores, label, color, fname, randomline=True):
    plt.figure(figsize=(4, 4), dpi=80)

    SetupROCCurvePlot(plt)
    AddROCCurve(plt, actual, scores, color, label)
    SaveROCCurvePlot(plt, fname, randomline)
