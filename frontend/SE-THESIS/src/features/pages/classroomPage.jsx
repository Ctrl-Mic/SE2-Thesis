import { useState, useEffect } from "react";
import { CirclePlus, ListVideo, Ellipsis, X } from "lucide-react";
import Popover from "@mui/material/Popover";
import { useForm } from "react-hook-form";
import { handleServerDown } from "../../shared/utils/serverDownHandler.js";
import { useServerStatus } from "../../context/serverStatusContext.jsx";
import { toast } from "sonner";
import { addRoom, getRooms } from "../../shared/services/roomService.js";
import { useNavigate } from "react-router-dom";
import { socket } from "../../shared/services/socketService.js";
import KebabPullout from "../../shared/components/ui/kebabPullout.jsx";
import EditClassroom from "../../shared/components/ui/editClassroom.jsx";
import EditSchedule from "../../shared/components/ui/editSchedule.jsx";
import ViewClassroom from "../../shared/components/ui/viewClassroom.jsx";

export default function classroomPage() {
  const navigate = useNavigate();
  const [rooms, setRooms] = useState([]);
  const [kebabAnchorEl, setKebabAnchorEl] = useState(null);
  const [selectedRoomId, setSelectedRoomId] = useState(null);
  const [selectedRoomName, setSelectedRoomName] = useState(null);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editScheduleModalOpen, setEditScheduleModalOpen] = useState(false);
  const [viewModalOpen, setViewModalOpen] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm();

  const { isServerUp, setIsServerUp } = useServerStatus();

  const onError = (errors) => {
    if (errors.cr_name) {
      toast.error(errors.cr_name.message);
    }
  };

  const handleCardClick = (roomId, roomName) => {
    setSelectedRoomId(roomId);
    setSelectedRoomName(roomName);
    setViewModalOpen(true);
  };

  const handleKebabClick = (event, roomId, roomName) => {
    event.stopPropagation();
    setKebabAnchorEl(event.currentTarget);
    setSelectedRoomId(roomId);
    setSelectedRoomName(roomName);
  };

  const handleKebabClose = () => {
    setKebabAnchorEl(null);
    setSelectedRoomId(null);
  };

  const handleEditClassroom = () => {
    setEditModalOpen(true);
  };

  const handleEditSchedule = () => {
    setEditScheduleModalOpen(true);
  };

  const onSubmit = async (data) => {
    try {
      const res = await addRoom({
        room_name: data.cr_name,
      });

      toast.success(res.message);
    } catch (error) {
      if (handleServerDown(error, setIsServerUp, navigate)) return;
      const message = error.response?.data?.message;
      toast.error(message);
    }
  };

  const [anchorEl, setAnchorEl] = useState(null);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  return (
    <div className="w-full h-full flex flex-col gap-2 bg-[#E4E3E1] min-h-0">
      <section className="relative w-full h-fit flex">
        <div className="w-full flex flex-row items-end justify-between text-[#4F4F4F]">
          <h1 className="text-subheader font-bold">Monitor Classroom</h1>
          <button
            onClick={handleClick}
            className="flex items-center gap-2 px-3 py-2 text-subtitle text-[#4F4F4F] hover:transition-all hover:scale-102 duration-300 cursor-pointer"
          >
            <CirclePlus size={50} color="#A1A2A6" />
            Add classroom
          </button>
        </div>
      </section>

      <section className="flex-1 overflow-y-auto min-h-0 grid grid-cols-4 gap-10 p-4">
        {rooms.map((room) => (
          <div
            key={room._id}
            onClick={() => handleCardClick(room._id, room.room_name)}
            className="relative overflow-clip bg-[#DFDEDA] shadow-outside-dropshadow aspect-3/2 rounded-lg hover:scale-101 duration-100 transition-all flex flex-col cursor-pointer"
          >
            <div className="bg-[#DFDEDA] w-full h-[70%] shadow-inner shadow-black/10 flex items-center justify-center">
              <ListVideo size={60} color="#A1A2A6" />
            </div>
            <div className="w-full flex-1 bg-[#C4C3C0] shadow-inner shadow-black/30 flex items-center justify-between px-4">
              <div>
                <p className="text-subtitle text-[#4F4F4F]">{room.room_name}</p>
              </div>
              <button
                onClick={(e) => handleKebabClick(e, room._id, room.room_name)}
                className="cursor-pointer hover:scale-110 transition-transform duration-150 p-2 rounded-full hover:bg-black/10"
              >
                <Ellipsis size={50} color="#A1A2A6" />
              </button>
            </div>
          </div>
        ))}
      </section>

      <Popover
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        slotProps={{
          paper: {
            sx: {
              backgroundColor: "#DFDEDA",
              color: "#A1A2A6",
              borderRadius: "15px",
            },
          },
        }}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
      >
        <div className="bg-[#DFDEDA] shadow-inner shadow-black/10 flex flex-col gap-2 items-center justify-center">
          <div className="w-[25vw] bg-[#C4C3C0] shadow shadow-black/20 flex-1 flex flex-row items-center justify-between p-5">
            <h2 className="text-subtitle text-[#4F4F4F]">Add Classroom</h2>
            <button
              onClick={handleClose}
              className="cursor-pointer hover:scale-105 transition-all duration-150"
            >
              <X />
            </button>
          </div>
          <div className="w-full px-5">
            <form
              onSubmit={handleSubmit(onSubmit, onError)}
              className="w-full flex flex-col gap-5 py-5 items-center justify-center"
            >
              <input
                {...register("cr_name", { required: "Please fill the form" })}
                className="w-full bg-[#E4E3E1] primary-text rounded-3xl px-6 py-4 shadow-inside-dropshadow-small font-light text-subtitle"
                placeholder="Enter Classroom Name"
              />
              <button
                type="submit"
                disabled={isSubmitting}
                className={`w-full bg-[#A1A2A6] text-subtitle text-[#E4E3E1] shadow-outside-dropshadow py-4 rounded-3xl`}
              >
                {isSubmitting ? "Adding..." : "Add Room"}
              </button>
            </form>
          </div>
        </div>
      </Popover>

      <KebabPullout
        open={Boolean(kebabAnchorEl)}
        anchorEl={kebabAnchorEl}
        onClose={handleKebabClose}
        onEditClassroom={handleEditClassroom}
        onEditSchedule={handleEditSchedule}
      />

      <EditClassroom
        open={editModalOpen}
        onClose={() => setEditModalOpen(false)}
        roomId={selectedRoomId}
      />

      <EditSchedule
        open={editScheduleModalOpen}
        onClose={() => setEditScheduleModalOpen(false)}
        roomId={selectedRoomId}
      />

      <ViewClassroom
        open={viewModalOpen}
        onClose={() => setViewModalOpen(false)}
        roomId={selectedRoomId}
        roomName={selectedRoomName}
      />
    </div>
  );
}
