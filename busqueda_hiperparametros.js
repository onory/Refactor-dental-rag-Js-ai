const brain = require('brain.js');

/**
 * Conjunto de datos históricos de entrenamiento (Consultorio Dra. Nancy Rivera)
 */
const datosEntrenamiento = [
  { input: { diaSemana: 0.1, turno: 0.2, tipoServicio: 1.0, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.12, requiereInstrumentalEspecial: 0.95 } },
  { input: { diaSemana: 0.1, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.91, requiereInstrumentalEspecial: 0.15 } },
  { input: { diaSemana: 0.2, turno: 0.2, tipoServicio: 0.7, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.18, requiereInstrumentalEspecial: 0.40 } },
  { input: { diaSemana: 0.2, turno: 0.8, tipoServicio: 0.7, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.85, requiereInstrumentalEspecial: 0.45 } },
  { input: { diaSemana: 0.3, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.05, requiereInstrumentalEspecial: 0.10 } },
  { input: { diaSemana: 0.3, turno: 0.2, tipoServicio: 0.2, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.94, requiereInstrumentalEspecial: 0.05 } },
  { input: { diaSemana: 0.4, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.92, requiereInstrumentalEspecial: 0.15 } },
  { input: { diaSemana: 0.4, turno: 0.2, tipoServicio: 1.0, esUrgencia: 1.0 }, output: { viabilidadAgenda: 0.65, requiereInstrumentalEspecial: 0.96 } },
  { input: { diaSemana: 0.5, turno: 0.2, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.89, requiereInstrumentalEspecial: 0.15 } },
  { input: { diaSemana: 0.6, turno: 0.8, tipoServicio: 0.4, esUrgencia: 0.0 }, output: { viabilidadAgenda: 0.02, requiereInstrumentalEspecial: 0.10 } },
];

// 1. Definición del Espacio de Búsqueda (Grid Search sobre Hiperparámetros)
const espacioBusqueda = {
  topologias: [
    [6, 4],     // Arquitectura base de la actividad anterior
    [8, 6],     // Mayor capacidad de representación
    [12, 8, 4], // Red profunda de 3 capas ocultas
  ],
  activaciones: ['sigmoid', 'leaky-relu', 'tanh'],
  tasasAprendizaje: [0.1, 0.3],
  momentums: [0.1, 0.5],
};

console.log('=========================================================================');
console.log(' BÚSQUEDA DE HIPERPARÁMETROS (GRID SEARCH) - AGENDA DRA. NANCY RIVERA');
console.log('=========================================================================\n');

let mejorConfiguracion = null;
let menorError = Infinity;
let redOptima = null;
let evaluacionId = 0;

// 2. Exploración sistemática del espacio de configuraciones
for (const hiddenLayers of espacioBusqueda.topologias) {
  for (const activation of espacioBusqueda.activaciones) {
    for (const learningRate of espacioBusqueda.tasasAprendizaje) {
      for (const momentum of espacioBusqueda.momentums) {
        evaluacionId++;

        const net = new brain.NeuralNetwork({
          hiddenLayers,
          activation,
        });

        const inicio = performance.now();
        const stats = net.train(datosEntrenamiento, {
          iterations: 5000,
          errorThresh: 0.004,
          learningRate,
          momentum,
          log: false,
        });
        const tiempoMs = (performance.now() - inicio).toFixed(2);

        const vectorActual = {
          id: evaluacionId,
          hiddenLayers,
          activation,
          learningRate,
          momentum,
          errorFinal: stats.error,
          iteraciones: stats.iterations,
          tiempoMs: `${tiempoMs} ms`,
        };

        console.log(
          `[Eval #${String(evaluacionId).padStart(2, '0')}] Capas: [${hiddenLayers}] | Act: ${activation.padEnd(10)} | LR: ${learningRate} | Mom: ${momentum} => Error: ${stats.error.toFixed(6)} (${stats.iterations} iters, ${tiempoMs} ms)`
        );

        // 3. Criterio de selección: menor error de convergencia (y desempate por menor número de iteraciones)
        if (
          stats.error < menorError ||
          (Math.abs(stats.error - menorError) < 1e-6 &&
            mejorConfiguracion &&
            stats.iterations < mejorConfiguracion.iteraciones)
        ) {
          menorError = stats.error;
          mejorConfiguracion = vectorActual;
          redOptima = net;
        }
      }
    }
  }
}

console.log('\n=========================================================================');
console.log(' VECTOR DE CONFIGURACIÓN ÓPTIMO ENCONTRADO');
console.log('=========================================================================');
console.log(JSON.stringify(mejorConfiguracion, null, 2));

// 4. Validación de inferencia con la red neuronal optimizada
console.log('\n=========================================================================');
console.log(' INFERENCIA CON EL MODELO OPTIMIZADO (DRA. NANCY RIVERA)');
console.log('=========================================================================');

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
  const prediccion = redOptima.run(caso.entrada);
  console.log(`\n${caso.descripcion}`);
  console.log(` -> Viabilidad de agendar bloque: ${(prediccion.viabilidadAgenda * 100).toFixed(2)}%`);
  console.log(` -> Requiere instrumental quirúrgico/especial: ${(prediccion.requiereInstrumentalEspecial * 100).toFixed(2)}%`);
});