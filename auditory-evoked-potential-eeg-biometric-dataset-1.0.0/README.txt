Abstract
This data set consists of over 240 two-minute EEG recordings obtained from 20 volunteers. Resting-state and auditory stimuli experiments are included in the data. The goal is to develop an EEG-based Biometric system.

The data includes resting-state EEG signals in both cases: eyes open and eyes closed. The auditory stimuli part consists of six experiments; Three with in-ear auditory stimuli and another three with bone-conducting auditory stimuli. The three stimuli for each case are a native song, a non-native song, and neutral music.

Background
The recoded dataset is multipurpose.  It is helpful for simple analysis of resting-state, comparison between resting state with eyes closed and with eyes open, studying the effect of auditory stimuli in terms of both methods of conduction by determining the neuronal mechanisms that follow the auditory stimulation and analysis of how different languages may affect the behaviour of human evoked potentials.

Additionally, it could be used for a simple EEG-based user authentication system.

The data was recorded at Marche Polytechnic University (UNIVPM) - Faculty of Engineering.

Recording Tools

OpenBCI Ganglion Board, 200 Hz sampling rate, four channels: T7, F8, Cz, and P4.
Gold Cup Electrodes.
Ten20 Conductive Paste.
Software: OpenBCI GUI · v5.0.3
Methods
Determination of eligibility

The subjects filled out a questionnaire that contains the subjects’ information. Additionally, the subjects read and agreed to a detailed informed consent.

Preparation and installation of equipment

The experiments start with the installation of the equipment needed. This includes four gold-cup electrodes with Ten20 Conductive Paste on the scalp. The electrodes were placed on the T7, F8, Cz, and P4 positions according to the 10/10 international EEG system [1], which were chosen according to these publications [2-4]. Additionally, two electrodes were placed in the left and right ears as a reference electrode and ground electrode, respectively. Once all of the electrodes were installed, a simple calibration was performed to ensure that everything worked correctly. The calibration included checking the connectivity of the electrodes by measuring the skin impedance. The The non-noisy two-minutes were segmented from the raw data.

Testing and data recording

The subject was asked to sit down and relax on a comfortable chair. The recording was performed in a single day per subject with the same order of tasks, A non-noisy two minutes recordings were segmented from the raw data. The recordings nvolves the acquisition of the electroencephalographic signal as follows:

Three minutes of resting-state, eyes open for three sessions.
Three minutes of resting-state, eyes closed for three sessions.
Non-Related experiment (Not provided in the dataset).
Three minutes of resting-state, eyes open for three sessions using noise isolation headset.
Non-Related experiment (Not provided in the dataset).
Three minutes of resting-state, eyes open for three sessions using noise isolation headset.
Three minutes of listing to a song in their native language using in-ear headphones.
Three minutes of listing to a song in a non-native language using in-ear headphones.
Three minutes of listing to neutral music using in-ear headphones.
Three minutes of listing to a song in their native language using bone-conducting headphones.
Three minutes of listing to a song in a non-native language using bone-conducting headphones.
Three minutes of listing to neutral music using bone-conducting headphones.
Note 1: If the person is Italian: the Arabic song was used for the non-Native song.

Note 2: If the person is not Italian: the Italian song was used for the non-Native song.

Note 3: Neutral music is a musical genre that emphasizes tranquillity, relaxation, and peaceful soundscapes. It is typically composed of instrumental music (music with no or wordless vocals), and it may involve acoustic instruments, electric synthesizers, and even recorded nature sounds.

Data Description
The Data are provided in .csv format. The recordings are named as follows:

sXX_exXX_sXX: s(Subject Number)_ex(Experiment Number)_s(Session Number. Just for ex01 and ex02).

Example: s03_ex02_s01: Subject 03 Experiment 02 Session

Each file contains five columns: Sample index and four EEG channels.

The Data is provided in three folders:

Raw Data: Each record contains the dataset as recorded without segmenting and without filtering.
Segmented Data: Each record contains two minutes of the EEG signals having the lowest possible noise, i.e. the data when the subject was moving and blinking their eyes less. Data were segmented manually, and the initial and final positions of the trim points are presented in the attached excel file: data_trim.csv.
Filtered Data: Each record contains two minutes of the EEG signals which includes the segmented data after applying a 1st order 1-40 Hz Butterworth filter and a 50 Hz notch filter with a quality factor of 30.
 

Data_trim.csv: Contains the initial and final point where the Segmented Data was trimmed from the Raw Data. The points are provided in seconds and samples.
Subjects.csv: Contains the metadata of the subjects and the questionnaire’s answers.
Songs. csv: Contains links for the songs used in the recordings
Usage Notes
The data was used in a Master's thesis at UNIVPM titled: Design and Implementation of Techniques for the Secure Authentication of Users Based on Electroencephalogram (EEG) Signals.

The reuse of the dataset involves, but is not limited to, studying the effect of the different auditory stimuli on EEG signals of both methods of conduction by determining the neuronal mechanisms that follow the auditory stimulus and analysis of how different languages may affect the behaviour of human evoked potentials. Additionally, studying the resting state differences between eyes open and eyes close states, comparing resting state with auditory stimuli cases. 

The limitation of the data is the low number of electrodes and subjects.

Acknowledgements
This data set was created and contributed to PhysioNet by Nibras Abo Alzahab (Nibras.Abo.Alzahab@gmail.com) and his colleagues, Angelo Di Iorio, Luca Apollonio, Muaaz Alshalak, Alessandro Gravina, Luca Antognoli, Lorenzo Scalise, Marco Baldi and Bilal Alchalabi.

The recordings were conducted at Marche Polytechnic University (UNIVPM) - Faculty of Engineering: Via Brecce Bianche, 12, 60131 Ancona AN, Italy. The subjects were asked to read and sign a detailed informed consent. The experiments were conducted in compliance with the World Medical Association (WMA) Declaration of Helsinki [5]. All the individually identifiable health information of the subjects and their identity are protected and can not be accessed with non-authorised users in order to preserve privacy for research participants.

Conflicts of Interest
The author(s) have no conflicts of interest to declare

References
Jurcak V, Tsuzuki D, Dan I. 10/20, 10/10, and 10/5 systems revisited: their validity as relative head-surface-based positioning systems. Neuroimage. 2007 Feb 15;34(4):1600-11.
Altahat SH. Robust EEG Channel Set for Biometric Application (Doctoral dissertation, University of Canberra).
Ravi KV, Palaniappan R. A minimal channel set for individual identification with EEG biometric using genetic algorithm. InInternational Conference on Computational Intelligence and Multimedia Applications (ICCIMA 2007) 2007 Dec 13 (Vol. 2, pp. 328-332). IEEE.
Marcel S, Millán JD. Person authentication using brainwaves (EEG) and maximum a posteriori model adaptation. IEEE transactions on pattern analysis and machine intelligence. 2007 Feb 20;29(4):743-52.
World Medical Association. World Medical Association Declaration of Helsinki. Ethical principles for medical research involving human subjects. Bulletin of the World Health Organization. 2001;79(4):373.