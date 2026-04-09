import React from "react";
import Logo from "@/assets/icons/logo_w.png";
import Dashboard from "@/assets/icons/home.png";
import Room from "@/assets/icons/dashboard.png";
import Activity from "@/assets/icons/activity.png";
import Notif from "@/assets/icons/notif.png";
import Dev from "@/assets/icons/dev.png";
import Organization from "@/assets/icons/orgGroup.png";
import Ex from "@/assets/icons/ex.png";

import { useAuth } from "../../../context/authContext";
import { useNavigate } from "react-router-dom";
import { useRooms } from "../../../context/roomContext";
import { useCamera } from "../../../context/cameraContext.jsx";
import { socket } from "../../services/socketService";
import { toast } from "sonner";
import SlideRight from "../animations/slideRight";

export default function Navbar() {
  const navigate = useNavigate();
  const { resetRooms } = useRooms();
  const { logout, user } = useAuth();
  const { stopAllCameras } = useCamera();

  return (
    <SlideRight>
      <div className="flex flex-col w-full h-full bg-[#A1A2A6] justify-between shadow-outside-dropshadow rounded-2xl p-4 gap-6">
        <div>
          <img src={Logo} alt="Logo" className="w-full" />
        </div>
        <SlideRight selector=".stagger" stagger={0.2} duration={0.5}>
          <div className="relative flex flex-col gap-5">
            <button
              onClick={() => navigate("/iris/home")}
              className="w-full aspect-square stagger bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
            >
              <img src={Dashboard} alt="Home" />
            </button>

            <button
              onClick={() => navigate("/iris/room_management")}
              className="w-full aspect-square stagger bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
            >
              <img src={Room} alt="Dashboard" />
            </button>

            <button
              onClick={() => navigate("/iris/activity")}
              className="w-full aspect-square stagger bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
            >
              <img src={Activity} alt="Activity" />
            </button>

            <button
              onClick={() => navigate("/iris/notifications")}
              className="w-full aspect-square stagger bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
            >
              <img src={Notif} alt="Notifications" />
            </button>
            
            {user?.is_admin || user?.is_authorized && (
              <button
                onClick={() => navigate("/iris/organization")}
                className="w-full aspect-square bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
              >
                <img src={Organization} alt="Organization" />
              </button>
            )}

            <button
              onClick={() => navigate("/iris/development")}
              className="w-full aspect-square stagger bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
            >
              <img src={Dev} alt="Notifications" />
            </button>
          </div>
        </SlideRight>
        <button
          onClick={() => {logout(), resetRooms(), stopAllCameras()}}
          className="w-full aspect-square bg-[#E4E3E1] rounded-full flex items-center shadow-outside-dropshadow-small justify-center cursor-pointer hover:scale-102 hover:bg-[#d4d3d1] transition-transform duration-300 tooltip"
        >
          <img src={Ex} alt="Logout" />
        </button>
      </div>
    </SlideRight>
  );
}
