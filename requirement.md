好的，现在来做下一步！我们已经完成了子录音的切分过程，接下来需要做ASR，我来指导你过程

[1]第一步
参照senseVoice-small的使用示例代码。model路径我已经配置好了。inference_pipeline传入的是需要处理的wav文件
```
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='/home/dzs-ai-4/dzs-dev/end2end_autio2kg/senseVoice-small',
    model_revision="master",
    device="cuda:0",)

rec_result = inference_pipeline('{传入的wav}')
print(rec_result)
```

按照现在的系统，每个wav文件都基于pyannote被处理成了一个文件夹下的多个子文件，并且都严格编号了顺序号。

[2]第二步
参照[1]中的示例代码，为每个文件夹的子文件顺序使用senseVoice-small做ASR，将结果写入docs/{xxx}.md中

格式要求
按照子录音顺序以及文件名字，将识别结果处理为说话人：ASR识别结果。举个例子

假设wavs/test1中有000_000_SPEAKER_01-0.031-1.398.wav和001_SPEAKER_00-1.027-2.444.wav

处理完第一个ASR后拼接到md中，即
SPEAKER_01:ASR结果
“这空一行”
SPEAKER_00:ASR结果

以此类推来完成一个wav的子wav的处理

[3]第三步
我先现在的系统都是批量处理（处理一个文件夹下的所有内容）实现这个新功能要兼容之前的设计