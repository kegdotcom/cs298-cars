import { distance_to_circle, distance_to_line, reflect } from "./math-functions.js";

let obstacleCt = 0;

class Wall {
  constructor (x1, y1, x2, y2) {
    this.id = obstacleCt++;
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
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
  constructor (x, y, vx, vy) {
    this.id = obstacleCt++;
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.theta = Math.atan2(vy, vx);
  }
  static radius = 10;

  draw(context) {
    context.save();
    // position context
    context.translate(this.x, this.y);
    // context.rotate(Math.atan2(this.vy, this.vx) + baseRotation);
    context.strokeStyle = "black";
    context.fillStyle = "blue";

    // inner dot
    context.moveTo(0, 0);
    context.beginPath();
    context.arc(0, 0, Car.radius/4, 0, 2*Math.PI);
    context.closePath();
    context.fill();
    // outer circle
    context.moveTo(0, 0);
    context.beginPath();
    context.arc(0, 0, Car.radius, 0, 2*Math.PI);
    context.closePath();
    context.stroke();
    // direction line
    context.beginPath();
    context.moveTo(0, 0);
    context.lineTo(this.vx * Car.radius, this.vy * Car.radius);
    context.stroke();
    
    // draw rays
    // const numRays = 5;
    // for (let i = 0; i < numRays; i++) {
    //   walls.forEach(wall => {
    //     //
    //   });
    //   cars.forEach(car => {
    //     //
    //   });
    //   context.strokeStyle = "red";
    //   context.moveTo(0, 0);
    //   context.rotate(-Math.PI/2);
    //   for (let i = 0; i < numRays; i++) {
    //     context.beginPath();
    //     context.moveTo(0, 0);
    //     context.lineTo(2*Car.radius, 0);
    //     context.stroke();
    //     context.rotate(Math.PI/4);
    //   }
    // }

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
  cars.push(new Car(getRandom(Car.radius, 500-Car.radius), getRandom(Car.radius, 500-Car.radius), getRandom(1, 10), getRandom(1, 10)));
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
    let startx = getRandom(0, 500-lengthx)
    let starty = getRandom(0, 500-lengthy)
    walls.push(new Wall(startx, starty, startx+lengthx, starty+lengthy));

  }

function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  context.strokeStyle = "gray";
  walls.forEach(wall => wall.draw(context));
  context.restore();
  cars.forEach(car => car.draw(context));
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
      const allObstacles = [...walls, ...cars.slice(0, i), ...cars.slice(i+1)];
      const obstacles = allObstacles.map(obstacle => ({
        obstacle: obstacle,
        distance: Math.abs(  
          obstacle instanceof Car ?
            distance_to_circle(car.x, car.y, obstacle.x, obstacle.y, Car.radius) :
            distance_to_line(car.x, car.y, obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2)
        )
      }));
      obstacles.sort((a, b) => a.distance - b.distance);
      obstacles.forEach(({obstacle, distance}) => {
        console.log(obstacle, distance)
        const collisionKey = `${car.id}-${obstacle.id}`;
        const cooldown = collisionCooldowns.get(collisionKey);
        if (cooldown > 0) {
          collisionCooldowns.set(collisionKey, cooldown-1);
          return;
        };
        if (distance <= Car.radius) {
          console.log("bam");
          collisionCooldowns.set(collisionKey, 3);
          let newVx, newVy;
          if (obstacle instanceof Car) {
            [newVx, newVy] = reflect(car.vx, car.vy, car.x - obstacle.x, car.y - obstacle.y);
            const overlap = 2*Car.radius - distance;
            car.x += (car.x - obstacle.x) * overlap / distance;
            car.y += (car.y - obstacle.y) * overlap / distance;
          } else {
            [newVx, newVy] = reflect(car.vx, car.vy, (obstacle.y1 - obstacle.y2), (obstacle.x2 - obstacle.x1));
          }
          console.log(newVx, newVy);
          car.vx = newVx;
          car.vy = newVy;
        }
      });
      car.x = Math.min(Math.max(car.x + car.vx * dTime * baseSpeed, Car.radius), 500-Car.radius);
      car.y = Math.min(Math.max(car.y + car.vy * dTime * baseSpeed, Car.radius), 500-Car.radius);
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