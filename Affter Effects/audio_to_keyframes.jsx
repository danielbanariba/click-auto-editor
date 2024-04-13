// Genera los keyframes de un audio  para que la imagen se pueda mover al ritmo de la musica

var comp = app.project.activeItem;
var audioLayer;

for (var i = 1; i <= comp.numLayers; i++) {
    var layer = comp.layer(i);
    if (layer.hasAudio) {
        audioLayer = layer;
        break;
    }
}

if (audioLayer) {
    audioLayer.selected = true;
    app.executeCommand(app.findMenuCommandId("Convert Audio to Keyframes"));
} else {
    alert("No se encontró ninguna capa de audio");
}