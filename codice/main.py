import weather_lib as wl


url = 'http://api.weatherbit.io/v2.0/history/hourly'


#wl.extract_frames('Eumetsat_View_2022-02-14_00_00-2024-02-14_00_00.mp4', 'frames')

frames=wl.save_frames('frames')
df=wl.show_frames('datiLarge.csv', frames)

wl.resize_dataset(df, 100, 50)

X_train, X_test, y_train, y_test = wl.prec_and_split(df['precip'], df['Images'])

wl.train_cnn(X_train, X_test, y_train, y_test, 50)