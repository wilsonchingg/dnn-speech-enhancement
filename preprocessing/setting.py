def init():
    global sample_sub_dir
    global voice_dir
    global voice_dev_dir
    global barkscale_16000_center
    global barkscale_16000
    voice_dir = 'LibriSpeech'
    voice_dev_dir = 'LibriSpeech_Dev'
    sample_sub_dir = []
    sample_sub_dir.append('DKITCHEN')
    sample_sub_dir.append('DLIVING')
    sample_sub_dir.append('DWASHING')
    sample_sub_dir.append('NFIELD')
    sample_sub_dir.append('NPARK')
    sample_sub_dir.append('NRIVER')
    sample_sub_dir.append('OHALLWAY')
    sample_sub_dir.append('OMEETING')
    sample_sub_dir.append('OOFFICE')
    sample_sub_dir.append('PCAFETER')
    sample_sub_dir.append('PRESTO')
    sample_sub_dir.append('PSTATION')
    sample_sub_dir.append('SCAFE')
    sample_sub_dir.append('SPSQUARE')
    sample_sub_dir.append('STRAFFIC')
    sample_sub_dir.append('TBUS')
    sample_sub_dir.append('TCAR')
    sample_sub_dir.append('TMETRO')
