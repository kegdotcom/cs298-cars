import { distance_to_circle, distance_to_line, reflect, ray_intersect_circle, ray_intersect_seg, getRandom } from "./math-functions.js";
import PolicyNetwork, { GlobalNetwork } from "./policynet.js";

let obstacleCt = 0;
class Wall {
  constructor({ p0, p1 } = {}) {
    this.id = ++obstacleCt;

    const [x0, y0] = p0 || [getRandom(0, 500), getRandom(0, 500)];
    const [x1, y1] = p1 || [getRandom(0, 500), getRandom(0, 500)];
    this.x1 = x0;
    this.y1 = y0;
    this.x2 = x1;
    this.y2 = y1;
    this.isCar = false;
  }

  draw(context) {
    context.save();
    context.strokeStyle = "gray";
    context.beginPath();
    context.moveTo(this.x1, this.y1);
    context.lineTo(this.x2, this.y2);
    context.stroke();
    context.restore();
  }
}

class Car {
  static radius = 10;
  static dTheta = 1;
  static dSpeed = 0.1;
  static epsilon = 0.75;
  static delta = 0.5;
  constructor({ x, y, vx, vy } = {}) {
    this.id = ++obstacleCt;
    this.x = x || getRandom(Car.radius, 500 - Car.radius);
    this.y = y || getRandom(Car.radius, 500 - Car.radius);
    this.vx = vx || getRandom(1, 10);
    this.vy = vy || getRandom(1, 10);
    this.collisions = 0;
    this.theta = Math.atan2(vy, vx);
    this.rayLengths = new Array(5);
    this.model = new PolicyNetwork(this.id, 10);
    this.isCar = true;
  }

  getState() {
    return [...this.rayLengths, this.x, this.y, this.vx, this.vy, this.collisions];
  }

  runNetworkFrame(train = false) {
    const stateTensor = tf.tensor2d(this.getState(), [1, this.model.STATE_SIZE]);
    const actionTensor = this.model.runFrame(stateTensor, train);
    const action = actionTensor.dataSync()[0];
    this.takeAction(action);
  }

  async runPretrainFrame(takeAction = false, log = false) {
    const state = this.getState();
    const stateTensor = tf.tensor2d(state, [1, this.model.STATE_SIZE]);
    const actionTensors = await this.model.trainOnPolicy([stateTensor], log);
    if (takeAction) {
      const action = actionTensors[0].dataSync()[0];
      this.takeAction(action);
    }
  }

  turnLeft() {
    const theta = -Car.dTheta * Math.PI / 180;
    const prevVX = this.vx;
    const prevVY = this.vy;
    this.vx = Math.cos(theta) * prevVX - Math.sin(theta) * prevVY;
    this.vy = Math.sin(theta) * prevVX + Math.cos(theta) * prevVY;
    this.slowDown(Car.epsilon);
  }

  turnRight() {
    const theta = Car.dTheta * Math.PI / 180;
    const prevVX = this.vx;
    const prevVY = this.vy;
    this.vx = Math.cos(theta) * prevVX - Math.sin(theta) * prevVY;
    this.vy = Math.sin(theta) * prevVX + Math.cos(theta) * prevVY;
    this.slowDown(Car.epsilon);
  }

  speedUp(factor = 1) {
    const speed = Math.hypot(this.vx, this.vy);
    const modifier = ((speed + Car.dSpeed) / speed)
    this.vx *= modifier * factor;
    this.vy *= modifier * factor;

    //this.vx = (1 / (1 - Car.dSpeed)) * (this.vx / speed * (speed + Car.dSpeed));
    //this.vy = (1 / (1 - Car.dSpeed)) * (this.vy / speed * (speed + Car.dSpeed));
  }

  slowDown(factor = 1) {
    const speed = Math.hypot(this.vx, this.vy);
    this.vx *= factor;
    this.vy *= factor;

    // this.vx = (1 - Car.dSpeed) * (this.vx / speed * (speed - Car.dSpeed));
    // this.vy = (1 - Car.dSpeed) * (this.vy / speed * (speed - Car.dSpeed));
  }

  takeAction(action, log = false) {
    if (log) {
      console.log(`Car ${this.id} took action ${action}`);
    }
    switch (action) {
      case 0:
      case 'L':
        this.turnLeft();
        break;
      case 1:
      case 'R':
        this.turnRight();
        break;
      case 2:
      case 'U':
        this.speedUp();
        break;
      case 3:
      case 'D':
        this.slowDown();
        break;
      default:
        break;
    }
  }


  draw(context, obstacles) {
    context.save();
    // position context
    context.translate(this.x, this.y);
    // context.rotate(Math.atan2(this.vy, this.vx) + baseRotation);
    context.strokeStyle = "black"; //this.id == 0 ? "blue" : "black";
    context.fillStyle = "blue";

    // inner dot
    context.moveTo(0, 0);
    context.beginPath();
    context.arc(0, 0, Car.radius / 4, 0, 2 * Math.PI);
    context.closePath();
    context.fill();
    // outer circle
    context.moveTo(0, 0);
    context.beginPath();
    context.arc(0, 0, Car.radius, 0, 2 * Math.PI);
    context.closePath();
    context.stroke();

    const sqrt1_2 = 1 / Math.sqrt(2);
    const rays = [
      [this.vy, -this.vx],
      [sqrt1_2 * (this.vx + this.vy), sqrt1_2 * (this.vy - this.vx)],
      [this.vx, this.vy],
      [sqrt1_2 * (this.vx - this.vy), sqrt1_2 * (this.vy + this.vx)],
      [-this.vy, this.vx],
    ];

    rays.forEach((ray, idx) => {
      let min_dist = 1e3;
      let min_dist_pt;
      obstacles.forEach(ob => {
        if (ob.id === this.id) return;
        let intersection = [];
        if (ob instanceof Car) {
          intersection = ray_intersect_circle([this.x, this.y], ray, [ob.x, ob.y], Car.radius);
        } else {
          intersection = ray_intersect_seg([this.x, this.y], ray, [ob.x1, ob.y1], [ob.x2, ob.y2]);
        }
        const intersection_dist = intersection ? Math.hypot(intersection[0] - this.x, intersection[1] - this.y) : 1000;
        if (intersection && intersection_dist < min_dist) {
          min_dist = intersection_dist;
          min_dist_pt = intersection;
        }
      });
      this.rayLengths[idx] = min_dist;
      if (min_dist_pt) {
        context.save();
        context.strokeStyle = "red";
        context.lineWidth = 0.5;
        context.beginPath();
        context.moveTo(0, 0);
        context.lineTo(min_dist_pt[0] - this.x, min_dist_pt[1] - this.y);
        context.stroke();
        context.restore();
      }
    });

    context.restore();
  }
}

async function draw(canvas, context, globalNetwork, frame, cars, walls = []) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.strokeStyle = "gray";
  walls.forEach(wall => wall.draw(context));
  cars.forEach(car => car.draw(context, [...cars, ...walls]));
  context.restore();

  const pretraining = frame < 1e4;
  console.log(`Frame ${frame}: ${pretraining ? "Pretraining" : "RL"}`)
  if (globalNetwork === null) {
    for (const car of cars) {
      if (pretraining) {
        // individual network pretraining
        await car.runPretrainFrame(true);
      } else {
        // individual network reinforce
        car.runNetworkFrame(true);
      }
    }
  } else {
    const stateTensors = cars.map(car => tf.tensor1d(car.getState()));
    const stateStack = tf.stack(stateTensors);
    const takeActions = true;
    let actions;
    if (pretraining) {
      // global network pretraining
      const actionsTensors = await globalNetwork.trainOnPolicy([stateStack])
      actions = actionsTensors[0].dataSync();
    } else {
      // global network reinforce
      const actionsTensor = globalNetwork.runFrame(stateStack);
      actionsTensor.print();
      actions = actionsTensor.dataSync();
    }

    if (takeActions) {
      cars.forEach((car, idx) => car.takeAction(actions[idx]));
    }
  }
}

function main(useGlobalNetwork = true, nCars = 3, nWalls = 2) {
  let paused = true;
  document.getElementById("play-pause").onclick = (e) => {
    paused = !paused;
    baseSpeed = baseSpeed === 0 ? 1 : 0;
    e.target.innerText = baseSpeed === 0 ? "Play" : "Pause";
    if (!paused) {
      window.requestAnimationFrame(loop);
    }
  }
  document.getElementById("speed").onchange = (e) => {
    baseSpeed = Number(e.target.value);
  }
  const canvas = document.getElementById("298-canvas");
  const context = canvas.getContext("2d");

  const cars = [];
  for (let c = 0; c < nCars; c++) {
    cars.push(new Car());
  }

  const walls = [
    new Wall({ p0: [0, 0], p1: [0, 500] }),
    new Wall({ p0: [0, 0], p1: [500, 0] }),
    new Wall({ p0: [0, 500], p1: [500, 500] }),
    new Wall({ p0: [500, 500], p1: [500, 0] }),
  ];
  for (let w = 0; w < nWalls; w++) {
    walls.push(new Wall());
  }

  const globalNetwork = useGlobalNetwork ? new GlobalNetwork(nCars, 10, 4) : null;
  const collisionCooldowns = new Map();
  const realWorldCooldownTime = 0.5; // wanted cooldown time in seconds 
  let frameRate = 60; // default frame rate
  let lastFrameTime = performance.now()
  let actionsTaken = 0;

  function loop(timestamp) {
    const dTime = lastFrameTime ? (timestamp - lastFrameTime) / 1000 : 0;
    frameRate = 1 / dTime; // Update frame rate dynamically
    lastFrameTime = timestamp;

    // Calculate the cooldown period in frames
    const cooldownInFrames = Math.round(realWorldCooldownTime * frameRate);

    for (let i = 0; i < cars.length; i++) {
      // get current car
      const car = cars[i];
      // get all obstacles except the current car
      const allObstacles = [...walls, ...cars.slice(0, i), ...cars.slice(i + 1)];
      const obstacles = allObstacles.map(obstacle => ({
        obstacle: obstacle,
        distance: Math.abs(
          obstacle instanceof Car ?
            distance_to_circle(car.x, car.y, obstacle.x, obstacle.y) - Car.radius :
            distance_to_line(car.x, car.y, obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2)
        )
      }));
      // obstacles.sort((a, b) => a.distance - b.distance);
      obstacles.forEach(({ obstacle, distance }) => {
        const collisionKey = `${car.id}-${obstacle.id}`;
        const cooldown = collisionCooldowns.get(collisionKey) || 0;
        if (cooldown > 0) {
          collisionCooldowns.set(collisionKey, cooldown - 1);
          return;
        };
        if (distance <= Car.radius) {
          console.log(`Car ${car.id} hit ${obstacle instanceof Car ? "Car" : "Wall"} ${obstacle.id}`);
          car.collisions += 1;
          collisionCooldowns.set(collisionKey, cooldownInFrames);
          let newVx, newVy;
          let obVx, obVy;
          if (obstacle instanceof Car) {
            [newVx, newVy] = reflect(car.vx, car.vy, car.x - obstacle.x, car.y - obstacle.y);
            [obVx, obVy] = reflect(obstacle.vx, obstacle.vy, obstacle.x - car.x, obstacle.y - car.y);
            const overlap = (Car.radius - distance) * 1.01;
            const theta = Math.atan2(car.y - obstacle.y, car.x - obstacle.x);
            car.x += overlap * Math.cos(theta);
            car.y += overlap * Math.sin(theta);
            obstacle.x -= overlap * Math.cos(theta);
            obstacle.y -= overlap * Math.sin(theta);
            obstacle.vx = obVx;
            obstacle.vy = obVy;

            // car.vx = newVx;
            // car.vy = newVy;
            obstacle.vx = obVx * Car.delta;
            obstacle.vy = obVy * Car.delta;
          } else {
            [newVx, newVy] = reflect(car.vx, car.vy, (obstacle.y1 - obstacle.y2), (obstacle.x2 - obstacle.x1));
            // car.vx = newVx;
            // car.vy = newVy;
          }
          car.vx = newVx * Car.delta;
          car.vy = newVy * Car.delta;
        }
      });
      // Update car position based on adjusted velocities
      car.x = Math.min(Math.max(car.x + car.vx * dTime * baseSpeed, Car.radius), 500 - Car.radius);
      car.y = Math.min(Math.max(car.y + car.vy * dTime * baseSpeed, Car.radius), 500 - Car.radius);
    }

    draw(canvas, context, globalNetwork, actionsTaken, cars, walls).then(() => {
      actionsTaken++;
      if (!paused) {
        window.requestAnimationFrame(loop);
      }
    });
  }
  window.requestAnimationFrame(loop);
}

/*
let baseRotation = 0;
document.addEventListener("keydown", e => {
  switch (e.key) {
    case "ArrowUp":
      baseSpeed += 0.1;
      break;
    case "ArrowDown":
      baseSpeed -= 0.1;
      break;
    case "ArrowLeft":
      baseRotation -= 0.1;
      break;
    case "ArrowRight":
      baseRotation += 0.1;
      break;
  }
})
*/

const nCars = 3;
const nWalls = 2;
let baseSpeed = 1;

const useGlobalNetwork = true;
main(useGlobalNetwork, nCars, nWalls);