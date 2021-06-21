from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

testy = [0]*40 + [1]*40
testycorr = [0]*10 + [1]*20
print(testy)
# ns_probs = [0.0002022406327,0.0004184260044,0.0003569202927,0.0007954605961,0.001096129464 ,0.000372172766
# ,0.0001508439663
# ,0.0003045817613
# ,0.0009670561281
# ,0.0003948242746,  0.001013341158
# ,9.92E-04
# ,9.24E-04
# ,8.46E-04
# ,9.49E-04
# ,9.60E-04
# ,1.01E-03
# ,1.07E-03
# ,9.22E-04
# ,1.06E-03 ]
gamma_2_3 = [
0.0006435154952
,0.0005056850562
,0.00100111096
,0.0008911993029
,0.01740253856
,0.001586874277
,0.001149744549
,0.0004435961028
,0.0008748878941
,0.0005891106424
,0.021190966572840725
,0.022980745010980394,0.021852390630011424
,0.022980745010980394
,0.022980745010980394
,0.021240219896123735
,0.021190966572840725
,0.02098170758701112
,0.0217661163438349
,0.021190966572840725
]
gamma_1_3 = [
0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.00011822734570709486
,0.001375395548020118
,0.000806469902322768
,0.0011421957191304387
,0.000806469902322768,0.000806469902322768,0.0008509305173593232
,0.0013753955480201184
,0.0009607502998290121
,0.0011499087535009832
,0.0013753955480201184
]
gauss = [
0.008386208936248436
,0.007939949245781742
,0.005798981892181397
,0.007939949245781742
,0.007939949245781742
,0.009995731744367369
,0.008386208936248436
,0.009482442502870616
,0.008620159851932943
,0.008386208936248436
,0.022346175367923455
,0.029350798176333237
,0.022797595853553645
,0.026690572212284727
,0.022078096001152122
,0.025727658306471713
,0.04040053673587386
,0.03563190306861846
,0.02680954753826731
,0.02238595633844215
]

uniform =  np.array([
0.00030243660054459013
,0.00250574294767443
,0.00017275101814793
,0.00027336187621368
,0.00054256084197077
,0.000298545501661458
,0.005204898766450558
,0.000199888695638192
,0.00022416623581908
,0.00690363792988304
,0.00260657559099969
,0.00273566851646961
,0.00260539062822649
,0.00283566851646961
,0.00363566851646961
,0.00460984159836949
,0.00560657559099969
,0.0063253505230707
,0.00463019629105045
,0.00560657559099969
])
gauss_med = [0.009622944413 ,0.008445598286 ,0.009088994376 ,0.01094908348 ,0.009990087532 ,0.008591987025 ,0.009354072288 ,0.03196462663 ,0.01673312441 ,0.008933013196 ,0.01049589898 ,0.00848001695 ,0.009473241335 ,0.008260977494 ,0.009115743059 ,0.0120867079 ,0.009554301256 ,0.01355554533 ,0.008526207485 ,0.01039298591 ,0.01022884699 ,0.00904631395 ,0.008544053425 ,0.01455119231 ,0.008125443686 ,0.008710000224 ,0.008630220551 ,0.009535576187 ,0.01046632594 ,0.01713284503 ,0.01097597426 ,0.008860480555 ,0.008785849201 ,0.008828382261 ,0.01522952899 ,0.008621035476 ,0.01031562485 ,0.009828843493 ,0.01109776361 ,0.009535576187 ,0.01061375929 ,0.02915513929 ,0.4871380203 ,0.05031512303 ,0.02755064548 ,0.04276743518 ,0.1698907761 ,0.9887120786 ,0.05865625439 ,0.04626715149 ,0.02999678909 ,0.05875694845 ,0.01353454804 ,0.340421787 ,0.01064645963 ,0.0106137559 ,0.02491825278 ,0.0481563819 ,0.1561064261 ,0.01838801106 ,0.0116647636 ,0.01085453721 ,0.01060970955 ,0.01217991272 ,0.1135802073 ,0.06608291059 ,0.16731576 ,0.02053700053 ,0.010611463 ,0.5331246992 ,0.03847143441 ,0.1182194603 ,0.1253461235 ,0.0720327598 ,0.1586167154 ,0.01061202462 ,0.122702107 ,0.02953754529 ,0.05540521903 ,0.04626715149]
notD_uniform = [0.06846597824 ,0.1167646594 ,0.2819581949 ,0.1167694421 ,0.1096752226 ,0.2823098151 ,0.110439022 ,0.1143380072 ,0.06846597824 ,0.1166078021 ,0.1855917596 ,0.1580348311 ,0.1340243376 ,0.1337313508 ,0.1110320219 ,0.116522958 ,0.1330266689 ,0.1167266637 ,0.1102388416 ,0.1105445493 ,0.1167868025 ,0.06846597824 ,0.06846597824 ,0.1167347551 ,0.116275144 ,0.06846597824 ,0.1097007303 ,0.1178425341 ,0.1860398025 ,0.1129131201 ,0.1168967682 ,0.1105358655 ,0.1165368566 ,0.1171258451 ,0.1170523778 ,0.1169455357 ,0.2819816964 ,0.1175359969 ,0.1128831664 ,0.1306493249 ,0.06846597824 ,0.1212680576 ,0.2045221486 ,0.1716125221 ,0.1030975717 ,0.1499912323 ,0.08625951243 ,0.279603035 ,0.06836069289 ,0.09663529684 ,0.1866613696 ,0.0808470913 ,0.2044281503 ,0.2042428866 ,0.08801609905 ,0.2286684744 ,0.2666699383 ,0.06753255358 ,0.07494370019 ,0.05935348837 ,0.1174874182 ,0.06845814087 ,0.0684601017 ,0.2722416591 ,0.204005685 ,0.06846597824 ,0.0618511475 ,0.2669553115 ,0.1874385506 ,0.270411231 ,0.131081994 ,0.267725074 ,0.1302694344 ,0.2395339915 ,0.1587531586 ,0.2114816771 ,0.08506949811 ,0.2078275145 ,0.0805337429 ,0.1669941951]
gamma_1_3_mean = [0.02906026632 ,0.01810496638 ,0.008341554377 ,0.01779171108 ,0.01633246172 ,0.01581073917 ,0.01887314583 ,0.01934041774 ,0.0162383882 ,0.04244496351 ,0.02267338135 ,0.009839083655 ,0.01672990381 ,0.01759921679 ,0.01453684022 ,0.01465851584 ,0.01678842122 ,0.01694775618 ,0.01523171887 ,0.01499388937 ,0.01222150224 ,0.1020067435 ,0.01414896418 ,0.002116956614 ,0.01704187404 ,0.01871845514 ,0.01644308284 ,0.005131670994 ,0.0148025059 ,0.06082090842 ,0.02222147183 ,0.01445372191 ,0.01023987724 ,0.01685966111 ,0.01365816156 ,0.01536930851 ,0.01430409825 ,0.06972655133 ,0.0102360524 ,0.0111285684 ,0.01389528644 ,0.1214036119 ,0.4138538398 ,0.237797672 ,0.593961425 ,0.005058691945 ,0.6834616625 ,0.4147349537 ,0.00831077186 ,0.3052739741 ,0.5294378023 ,0.02943171573 ,0.3991702553 ,0.01584998366 ,0.3354670917 ,0.06437848686 ,0.008665718867 ,0.4190501391 ,0.2560619727 ,0.0510188137 ,0.0595271248 ,0.7443969994 ,0.1060177925 ,0.6330727734 ,0.5245543365 ,0.3229398243 ,0.5250651725 ,0.1465471313 ,0.05379225896 ,0.1593660766 ,0.1483255726 ,0.3822447578 ,0.2651162213 ,0.3184481274 ,0.4162856997 ,0.009501095499 ,0.1567645953 ,0.01511336416 ,0.33619782 ,0.02995347662]
# print(len(gauss_mean))
# uniform = (uniform-np.min(uniform))/(np.max(uniform)-np.min(uniform))
print(uniform)
corr = [0.2938
        ,0.292 , 0.2941, 0.2936, 0.2923, 0.2937, 0.294 , 0.2934 ,0.2908 ,0.295
 ,0.9978, 0.9988 ,0.9995, 1. ,    0.9978, 1.0014 ,1.0025, 1.0003 ,1.0016 ,1.0041
 ,0.294,  0.2924, 0.2945, 0.2939, 0.2927, 0.294 , 0.2944, 0.2938, 0.2911, 0.2952
        ]
# calculate roc curves
ns_fpr, ns_tpr, thres = roc_curve(testy, notD_uniform )
# ns_fpr, ns_tpr, thres = roc_curve(testycorr, corr )
# ns_fprcorr, ns_tprcorr, thres = roc_curve(testycorr, corr )
# ns_fprgauss_med, ns_tprgauss_med, thres = roc_curve(testy, gauss_med )
# ns_fprnotD_uniform, ns_tprnotD_uniform, thres = roc_curve(testy, notD_uniform )
# ns_fprgamma_1_3_mean, ns_tprgamma_1_3_mean, thres = roc_curve(testy, gamma_1_3_mean )
# lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# print(ns_fpr)
# print(ns_tpr)
# print(thres)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr,'k' ,linestyle='-',marker='o', markersize=6, label='No Skill')
# pyplot.plot(ns_fprcorr, ns_tprcorr,'k' ,linestyle='-',marker='o', markersize=6, label='corr')
# pyplot.plot(ns_fprgauss_med, ns_tprgauss_med,'k' ,linestyle='--',marker='^', markersize=6, label='Our')
# pyplot.plot(ns_fprnotD_uniform, ns_tprnotD_uniform,'k' ,linestyle=':',marker='*', markersize=6, label='notD')
# pyplot.plot(ns_fprgamma_1_3_mean, ns_tprgamma_1_3_mean,'k' ,linestyle='-.',marker='d', markersize=6, label='sample')
# pyplot.plot(ns_fpr, ns_tpr, linestyle=':', label='No Skill')
# pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
# pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
pyplot.xlabel('False Positive Rate', fontsize=22)
pyplot.ylabel('True Positive Rate', fontsize=22)
# show the legend
# pyplot.legend()
# show the plot
# pyplot.show()
# pyplot.savefig("roc_gauss_med.png", bbox_inches='tight')
pyplot.savefig("roc_uniform_notD.png", bbox_inches='tight')
# pyplot.savefig("roc_gamma_1_3_mean.png", bbox_inches='tight')
# pyplot.savefig("roc_corr.png", bbox_inches='tight')
# pyplot.savefig("roc_all.png")

# skplt.metrics.plot_roc_curve(testy, uniform)
# plt.show()
