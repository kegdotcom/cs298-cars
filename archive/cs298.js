import { distance_to_circle, distance_to_line, reflect, ray_intersect_circle, ray_intersect_seg } from "../math-functions.js";
import PolicyNetwork from "../model.js";

let obstacleCt = 0;
const sqrt1_2 = 1 / Math.sqrt(2);
class Wall {
  constructor(x1, y1, x2, y2) {
    this.id = obstacleCt++;
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }

  getModelInput() {
    return tf.tensor
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
  static epsilon = 0.1;
  static delta = 0.5;
  constructor(x, y, vx, vy) {
    this.id = obstacleCt++;
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.theta = Math.atan2(vy, vx);
    this.rayLengths = new Array(5);
    this.model = new PolicyNetwork();
  }

  getState() {
    return [...this.rayLengths, this.x, this.y, this.vx, this.vy];
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
    const modifier = ((speed - Car.dSpeed) / speed)
    this.vx *= modifier * factor;
    this.vy *= modifier * factor;

    // this.vx = (1 - Car.dSpeed) * (this.vx / speed * (speed - Car.dSpeed));
    // this.vy = (1 - Car.dSpeed) * (this.vy / speed * (speed - Car.dSpeed));
  }

  takeAction(action) {
    switch (action) {
      case 'L':
        this.turnLeft();
        break;
      case 'R':
        this.turnRight();
        break;
      case 'U':
        this.speedUp();
        break;
      case 'D':
        this.slowDown();
        break;
      default:
        break;
    }
  }
  updateModelAction() {
    const carState = this.getState();
    const [action, prob] = this.model.predictActionProbs(carState);
    console.log(`Action taken: ${action} with probability: ${prob}`);
    this.takeAction(action);
  }
  handleCollision() {
    const speed = Math.hypot(this.vx, this.vy);
    const slowdownFactor = Car.delta / speed;
    this.vx = slowdownFactor * this.vx;
    this.vy = slowdownFactor * this.vy;
  }

  draw(context) {
    context.save();
    // position context
    context.translate(this.x, this.y);
    // context.rotate(Math.atan2(this.vy, this.vx) + baseRotation);
    context.strokeStyle = this.id == 0 ? "blue" : "black";
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

    const rays = [
      [this.vy, -this.vx],
      [sqrt1_2 * (this.vx + this.vy), sqrt1_2 * (this.vy - this.vx)],
      [this.vx, this.vy],
      [sqrt1_2 * (this.vx - this.vy), sqrt1_2 * (this.vy + this.vx)],
      [-this.vy, this.vx],
    ];

    rays.forEach((ray, idx) => {
      const obstacles = [...cars, ...walls];
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

const canvas = document.getElementById("298-canvas");
const context = canvas.getContext("2d");
let baseSpeed = 1;
let baseRotation = 0;

const numCars = 4;
const cars = [];
for (let i = 0; i < numCars; i++) {
  cars.push(new Car(getRandom(Car.radius, 500 - Car.radius), getRandom(Car.radius, 500 - Car.radius), getRandom(1, 10), getRandom(1, 10)));
}

const numWalls = 2;

const walls = [
  new Wall(0, 0, 0, 500),
  new Wall(0, 0, 500, 0),
  new Wall(0, 500, 500, 500),
  new Wall(500, 500, 500, 0),
]

for (let i = 0; i < numWalls; i++) {
  let lengthx = getRandom(100, 200)
  let lengthy = getRandom(100, 200)
  let startx = getRandom(0, 500 - lengthx)
  let starty = getRandom(0, 500 - lengthy)
  walls.push(new Wall(startx, starty, startx + lengthx, starty + lengthy));
}

let actionsTaken = 0;
function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.strokeStyle = "gray";
  walls.forEach(wall => wall.draw(context));
  context.restore();
  cars.forEach(car => {
    car.draw(context);
    if (car.id == 0) {
      car.updateModelAction();
    }
  });
}

function main() {
  const collisionCooldowns = new Map();
  let lastTime;
  function loop(timestamp) {
    const dTime = lastTime ? (timestamp - lastTime) / 1000 : 0;
    lastTime = timestamp;
    for (let i = 0; i < cars.length; i++) {
      // get current car
      const car = cars[i];
      // get all obstacles except the current car
      const allObstacles = [...walls, ...cars.slice(0, i), ...cars.slice(i + 1)];
      const obstacles = allObstacles.map(obstacle => ({
        obstacle: obstacle,
        distance: Math.abs(
          obstacle instanceof Car ?
            distance_to_circle(car.x, car.y, obstacle.x, obstacle.y, Car.radius) :
            distance_to_line(car.x, car.y, obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2)
        )
      }));
      // obstacles.sort((a, b) => a.distance - b.distance);
      obstacles.forEach(({ obstacle, distance }) => {
        const collisionKey = `${car.id}-${obstacle.id}`;
        const cooldown = collisionCooldowns.get(collisionKey);
        if (cooldown > 0) {
          collisionCooldowns.set(collisionKey, cooldown - 1);
          return;
        };
        if (distance <= Car.radius) {
          console.log("bam");
          collisionCooldowns.set(collisionKey, 3);
          let newVx, newVy;
          let obVx, obVy;
          if (obstacle instanceof Car) {
            [newVx, newVy] = reflect(car.vx, car.vy, car.x - obstacle.x, car.y - obstacle.y);
            [obVx, obVy] = reflect(obstacle.vx, obstacle.vy, obstacle.x - car.x, obstacle.y - car.y);
            const overlap = Car.radius - distance / 2;
            const theta = Math.atan2(car.y - obstacle.y, car.x - obstacle.x);
            car.x += overlap * Math.cos(theta);
            car.y += overlap * Math.sin(theta);
            obstacle.x -= overlap * Math.cos(theta);
            obstacle.y -= overlap * Math.sin(theta);
            obstacle.vx = obVx;
            obstacle.vy = obVy;
          } else {
            [newVx, newVy] = reflect(car.vx, car.vy, (obstacle.y1 - obstacle.y2), (obstacle.x2 - obstacle.x1));
          }
          car.vx = newVx;
          car.vy = newVy;
        }
      });
      car.x = Math.min(Math.max(car.x + car.vx * dTime * baseSpeed, Car.radius), 500 - Car.radius);
      car.y = Math.min(Math.max(car.y + car.vy * dTime * baseSpeed, Car.radius), 500 - Car.radius);
    }
    draw();
    window.requestAnimationFrame(loop);
  }
  window.requestAnimationFrame(loop);
}

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

document.getElementById("play-pause").onclick = (e) => {
  baseSpeed = baseSpeed === 0 ? 1 : 0;
  e.target.innerText = baseSpeed === 0 ? "Play" : "Pause";
}

document.getElementById("speed").onchange = (e) => {
  baseSpeed = Number(e.target.value);
}
function getRandom(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

main();