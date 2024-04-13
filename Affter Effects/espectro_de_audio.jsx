// Asume que 'comp' es tu composición y 'audioLayer' es la capa de audio
var audioLayer = comp.layer(2); // Asume que la capa de audio es la segunda capa

// Crea una nueva capa de espectro de audio
var spectrumLayer = comp.layers.addSolid([0, 0, 0], "Audio Spectrum", comp.width, comp.height, 1);

// Añade el efecto de espectro de audio a la capa
var spectrumEffect = spectrumLayer.property("Effects").addProperty("Audio Spectrum");

// Configura las propiedades del efecto de espectro de audio
spectrumEffect.property("Audio Layer").setValue(audioLayer.index);
spectrumEffect.property("Start Frequency").setValue(700);
spectrumEffect.property("End Frequency").setValue(9100);
spectrumEffect.property("Start Point").setValue([864.0, 2060.0]);
spectrumEffect.property("End Point").setValue([864.0, 32]);
spectrumEffect.property("Frequency bands").setValue(760);
spectrumEffect.property("Maximum Height").setValue(32000);
spectrumEffect.property("Thickness").setValue(7);
spectrumEffect.property("Softness").setValue(0); // 0%
spectrumEffect.property("Display Options").setValue(1); // Digital
spectrumEffect.property("Side Options").setValue(1); // Side A