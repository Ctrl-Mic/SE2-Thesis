// !Libraries
import { useState, useEffect } from "react";
// !Componenets
import { checkFirstTime } from "../../shared/services/authService";
import { socket } from "../../shared/services/socketService.js";
// import { getRooms } from "../../shared/services/roomService.js";
import { useRooms } from "../../context/roomContext.jsx";

export default function dashboard() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showGuide, setShowGuide] = useState(false);
  const { rooms } = useRooms();
  const emptyRooms = rooms.filter((room) => room.room_occupants > 0).length;
  const vacantRoom = rooms.filter((room) => room.room_occupants === 0).length;
  socket.on("connect", () => console.log("Socket connected!", socket.id));
  
  useEffect(() => {
    // getRooms();
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    // Listen for new rooms in real-time
    const handleNewRoom = (room) => setRooms((prev) => [...prev, room]);
    socket.on("roomAdded", handleNewRoom);

    return () => {
      socket.off("roomAdded", handleNewRoom);
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="w-full h-full bg-[#E4E3E1]">
      <section className="relative w-full h-full flex flex-col gap-6">
        <div className="w-full flex flex-row items-end justify-between text-[#1E1E1E] opacity-75">
          <h1 className="text-subheader font-bold">Dashboard</h1>
          <h2 className="text-subtitle">
            {currentTime.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}{" "}
            |{" "}
            {currentTime.toLocaleDateString([], {
              month: "short",
              day: "2-digit",
              year: "numeric",
            })}
          </h2>
        </div>

        <div className="relative w-full h-[30vh] flex flex-row items-center justify-start gap-6 font-bold">
          <div className="flex flex-col items-start gap-4 primary-text">
            <h2 className="text-title">Occupied Rooms</h2>
            <div className="w-[20vw] aspect-video rounded-2xl shadow-outside-dropshadow flex justify-center items-center text-header">
              {emptyRooms}
            </div>
          </div>
          <div className="flex flex-col items-start gap-4 primary-text">
            <h2 className="text-title">Vacant Rooms</h2>
            <div className="w-[20vw] aspect-video rounded-2xl shadow-outside-dropshadow flex justify-center items-center text-header">
              {vacantRoom}
            </div>
          </div>
          <div className="flex flex-col w-full h-full items-start gap-4 primary-text">
            <h2 className="text-title">Activity History</h2>
            <div className="w-full h-full rounded-2xl shadow-outside-dropshadow"></div>
          </div>
        </div>

        <div className="relative w-full h-full flex flex-col gap-4">
          <h1 className="text-title primary-text font-bold">
            Classroom Status
          </h1>
          <div className="w-full h-full rounded-2xl shadow-outside-dropshadow"></div>
        </div>
      </section>
    </div>
  );
}
