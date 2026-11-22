const brain = require('brain.js');

// 1. Configuración de la Red Neuronal
const net = new brain.NeuralNetwork({
// 2 capas ocultas (6 y 4 neuronas) con función  activa sigmoide    
  hiddenLayers: [6, 4],
  activation: 'sigmoid',
});

/**
 * 2. Conjunto de datos  entrenamiento embendding
 * Entradas normalizadas [0 a 1]:
 * - diaSemana: 0.1 (Lun), 0.2 (Mar), 0.3 (Mié), 0.4 (Jue), 0.5 (Vie), 0.6 (Sáb)
 * - turno: 0.2 (Mañana 09-13h), 0.8 (Tarde 15-19h)
 * - tipoServicio: 0.2 (Valoración 30m), 0.4 (Limpieza 45m), 0.7 (Resina 60m), 1.0 (Cirugía/Endodoncia 120m)
 * - esUrgencia: 0.0 (No), 1.0 (Sí)
 *
 * Salidas esperadas [0 a 1]:
 * - viabilidadAgenda: Probabilidad de disponibilidad inmediata según historial de ocupación
 * - requiereQuirofanoEspecial: Probabilidad de requerir preparación de instrumental quirúrgico
 */
const datosEntrenamiento = [
  // Lunes mañana ocupado (cirugía tercer molar) -> baja viabilidad sin urgencia
  { input: { diaSemana: 0.1, turno: 0.2, tipoServicio: 1.0, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.12, requiereInstrumentalEspecial: 0.95 } },
  // Lunes tarde disponible -> alta viabilidad para limpieza o valoración
  { input: { diaSemana: 0.1, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.91, requiereInstrumentalEspecial: 0.15 } },
  // Martes mañana ocupado (ortodoncia) -> baja viabilidad
  { input: { diaSemana: 0.2, turno: 0.2, tipoServicio: 0.7, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.18, requiereInstrumentalEspecial: 0.40 } },
  // Martes tarde bloque 15-17h disponible para resinas
  { input: { diaSemana: 0.2, turno: 0.8, tipoServicio: 0.7, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.85, requiereInstrumentalEspecial: 0.45 } },
  // Miércoles tarde ocupado (jornada académica Dra. Nancy Rivera) -> disponibilidad casi nula
  { input: { diaSemana: 0.3, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.05, requiereInstrumentalEspecial: 0.10 } },
  // Miércoles mañana disponible
  { input: { diaSemana: 0.3, turno: 0.2, tipoServicio: 0.2, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.94, requiereInstrumentalEspecial: 0.05 } },
  // Jueves tarde bloque disponible para limpieza dental
  { input: { diaSemana: 0.4, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.92, requiereInstrumentalEspecial: 0.15 } },
  // Jueves mediodía endodoncia (instrumental especial)
  { input: { diaSemana: 0.4, turno: 0.2, tipoServicio: 1.0, esUrgencia: 1.0 }, output: { viabilidadAgenda: 0.65, requiereInstrumentalEspecial: 0.96 } },
  // Viernes mañana disponible
  { input: { diaSemana: 0.5, turno: 0.2, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.89, requiereInstrumentalEspecial: 0.15 } },
  // Sábado tarde (fuera de horario, cierra 14:00 hrs) -> nulo
  { input: { diaSemana: 0.6, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.02, requiereInstrumentalEspecial: 0.10 } },
];

console.log('=================================================================');
console.log(' ENTRENAMIENTO RED NEURONAL - CONSULTORIO DRA. NANCY RIVERA');
console.log('=================================================================');

// 3. Ejecución del entrenamiento con Backpropagation
const stats = net.train(datosEntrenamiento, {
  iterations: 5000,
  errorThresh: 0.005,
  learningRate: 0.3,
  log: true,
  logPeriod: 1000,
});

console.log('\n>>> Resultado final del entrenamiento:', stats);

// 4. Inferencia / Predicción sobre nuevos escenarios de pacientes
console.log('\n=================================================================');
console.log(' INFERENCIA SOBRE NUEVAS SOLICITUDES DE CITA');
console.log('=================================================================');

const casosPrueba = [
  {
    descripcion: 'Caso 1: Paciente solicita Limpieza Dental el Jueves por la tarde (sin urgencia)',
    entrada: { diaSemana: 0.4, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 },
  },
  {
    descripcion: 'Caso 2: Paciente solicita Resina el Miércoles por la tarde (jornada académica)',
    entrada: { diaSemana: 0.3, turno: 0.8, tipoServicio: 0.7, esUrgencia: 0.0 },
  },
  {
    descripcion: 'Caso 3: Paciente con Urgencia de Endodoncia el Jueves por la mañana',
    entrada: { diaSemana: 0.4, turno: 0.2, tipoServicio: 1.0, esUrgencia: 1.0 },
  },
];

casosPrueba.forEach((caso) => {
  const prediccion = net.run(caso.entrada);
  console.log(`\n${caso.descripcion}`);
  console.log(` -> Viabilidad de agendar bloque: ${(prediccion.viabilidadAgenda * 100).toFixed(2)}%`);
  console.log(` -> Requiere instrumental quirúrgico/especial: ${(prediccion.requiereInstrumentalEspecial * 100).toFixed(2)}%`);
});