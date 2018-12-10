naobo
trumpet
cello

SD.downloadSoundsFreesound(queryText='naobo', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='testDownload/', topNResults=20, duration=(0,10))
downloadSoundsFreesound(queryText='trumpet', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='testDownload/', topNResults=20, duration=(0,3), tag='single-note')
downloadSoundsFreesound(queryText='cello', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='testDownload/', topNResults=20, duration=(0,3), tag='single-note')
SD.downloadSoundsFreesound(queryText='mridangam',tag='mridangam-stroke-dataset', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='testDownload/', topNResults=20, duration=(0,3))



SA.descriptorPairScatterPlot('testDownload/', descInput=(0,3))
SA.clusterSounds('testDownload/', 3, [0,3])
SA.clusterSounds('testDownload/', 3, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])


trumpet
SD.downloadSoundsFreesound(queryText='guitar', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='newDownload/', topNResults=1, duration=(0,3), tag='single-note')
SD.downloadSoundsFreesound(queryText='trumpet', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='newDownload/', topNResults=1, duration=(0,3))
SD.downloadSoundsFreesound(queryText='crash', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='newDownload/', topNResults=1, duration=(0,3))
SD.downloadSoundsFreesound(queryText='cello bow', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='newDownload/', topNResults=1, duration=(0,3))
SD.downloadSoundsFreesound(queryText='bell-ride', API_Key='s9xV21k899aPetFLLLj7IQjq2hCprXG0IqymUEy1', outputDir='newDownload/', topNResults=1, duration=(0,3))
cello bow


SA.classifySoundkNN('newDownload/guitar/91199/91199_1075352-lq.json','testDownload/', 3, descInput=[0,3])
SA.classifySoundkNN('newDownload/guitar/91199/91199_1075352-lq.json','testDownload/', 3, descInput=[0,3,11,12,13,14,15,16])
SA.classifySoundkNN('newDownload/trumpet/81822/81822_16052-lq.json','testDownload/', 3, descInput=[0,3])
SA.classifySoundkNN('newDownload/trumpet/81822/81822_16052-lq.json','testDownload/', 3, descInput=[0,3,11,12,13,14,15,16])
SA.classifySoundkNN('newDownload/crash/344266/344266_5121236-lq.json','testDownload/', 3, descInput=[0,3])
SA.classifySoundkNN('newDownload/crash/344266/344266_5121236-lq.json','testDownload/', 3, descInput=[0,3,11,12,13,14,15,16])
SA.classifySoundkNN('newDownload/cello-bow/68452/68452_649468-lq.json','testDownload/', 3, descInput=[0,3])
SA.classifySoundkNN('newDownload/bell-ride/41941/41941_433684-lq.json','testDownload/', 3, descInput=[0,3])


