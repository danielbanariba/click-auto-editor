// Obtén la capa "Audio Amplitude" y la capa de la imagen
var comp = app.project.activeItem;
var audioAmplitudeLayer = comp.layer("Audio Amplitude");

if (audioAmplitudeLayer === null) {
    throw new Error("No se encontró la capa 'Audio Amplitude'");
}

var imageLayer = comp.layer(comp.numLayers);

// Accede a la propiedad "Slider" en "Both Channels" en los efectos de la capa "Audio Amplitude"
var slider = audioAmplitudeLayer.effect("Both Channels")("Slider");

// Accede a la propiedad "Scale" de la capa de la imagen
var scale = imageLayer.property("Scale");

// Vincula la propiedad "Scale" de la imagen a la propiedad "Slider" del audio
scale.expression = "temp = thisComp.layer('Audio Amplitude').effect('Both Channels')('Slider')/3; [temp, temp]+[100, 100];";

// Accede a la propiedad "rotation" de la capa de la imagen
var rotation = imageLayer.property("rotation");

// pone el valor de la rotación 
rotation.expression = "wiggle(3,3)";