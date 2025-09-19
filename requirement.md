# 项目概述
这是一个端到端的客服对话wav录音转高质量知识库语料的系统。目前先只考虑处理单个录音，之后才考虑批量的处理和并发

一切的操作先在download_puannnote.py中完成，最小化更改，不要冗余

## 项目流水线
[1]使用pyannote.audio分离不同时间戳的说话人，并将信息写入rttm文件（已完成，已实现模型加载、提前载入内存与gpu处理加速、进度监控、写入attm）

[2]根据时间戳，使用torchaudio切分子录音，以下我给你详细描述这个过程（你需要做）

rttm的内容示例如下,更具体的可以查看rttms/111.rttm
```
SPEAKER waveform 1 0.031 2.008 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER waveform 1 1.027 0.354 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER waveform 1 2.039 0.388 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER waveform 1 2.427 0.017 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER waveform 1 3.457 1.569 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER waveform 1 5.313 0.017 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER waveform 1 5.330 0.709 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER waveform 1 6.038 0.473 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER waveform 1 6.511 0.456 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER waveform 1 7.439 0.017 <NA> <NA> SPEAKER_00 <NA> <NA>
```

第一、二列不用管，第三列为起始时间戳，第三列为持续时间，第五第六列不用管，第七列是pyannote.audio标识的说话人，第八、九列不用管

- 根据说话人、起始时间、持续时间来使用torchaudio切分音频。每个文件，例如wavs/111.wav切分后的结果放在wavs/111/下，每个文件命名为说话人-起始时间-结束时间（起始时间+持续时间）
- 切分的起始时间为rttm的起始时间，结束时间为（起始时间+持续时间）
- 因为是基于pyannote之上，分离出的时间戳效果严重受pyannote的影响，可能前后时间段会有重叠，你不需要理会！要求一行一行划分子录音就好

[3] 
