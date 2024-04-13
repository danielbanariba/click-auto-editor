//! NO FUNCIONA!

// Obtén el proyecto activo y la composición activa
var project = app.project;
var comp = project.activeItem;

// Crea una nueva capa de partículas
var particleLayer = comp.layers.addSolid([1, 1, 1], "Particles", comp.width, comp.height, 1);

// Añade el efecto de partículas a la capa
var particleEffect = particleLayer.property("Effects").addProperty("CC Particle Systems II");

// Configura las propiedades del efecto de partículas para que parezca nieve
particleEffect.property("Producer::Radius X").setValue(comp.width);
particleEffect.property("Producer::Radius Y").setValue(0);
particleEffect.property("Birth Rate").setValue(4);
particleEffect.property("Longevity (sec)").setValue(3);
particleEffect.property("Physics").property("Velocity").setValue(0.1);
particleEffect.property("Physics").property("Gravity").setValue(0.1);
particleEffect.property("Particle").property("Particle Type").setValue(2); // 2 = Faded Sphere
particleEffect.property("Particle").property("Birth Size").setValue(0.1);
particleEffect.property("Particle").property("Death Size").setValue(0.1);
particleEffect.property("Particle").property("Size Variation").setValue(0.5);
particleEffect.property("Particle").property("Max Opacity").setValue(100);