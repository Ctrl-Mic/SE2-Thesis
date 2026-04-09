import React from "react";
import { ChevronLeft } from "lucide-react";
import changeProfile from "@/assets/icons/changeProfile.png";
import personalInfo from "@/assets/icons/personalInfo.png";
import applyOrg from "@/assets/icons/applyOrg.png";
import logout from "@/assets/icons/logout.png";
import personalInfo_w from "@/assets/icons/personalInfo_w.png";
import applyOrg_w from "@/assets/icons/applyOrg_w.png";
import logout_w from "@/assets/icons/logout_w.png";

const NAV_ITEMS = [
  { key: "personal", label: "Personal Information", icon: personalInfo, iconActive: personalInfo_w },
  { key: "apply",    label: "Apply organization",   icon: applyOrg,     iconActive: applyOrg_w },
  { key: "logout",   label: "Logout",               icon: logout,       iconActive: logout_w },
];

export default function ProfileSidebar({ activeTab, onTabChange, user }) {
  return (
    <div className="relative w-[18vw] h-full aspect-video rounded-2xl shadow-outside-dropshadow flex flex-col justify-start items-center text-header pt-4 px-4">
      <button
        onClick={() => window.history.back()}
        className="absolute top-4 left-4 w-10 h-10 bg-transparent rounded-full flex items-center justify-center cursor-pointer hover:bg-[#d4d3d1] transition-colors duration-300"
      >
        <ChevronLeft size={30} className="text-[#858585]" />
      </button>
      <div className="w-40 aspect-square rounded-full flex items-center justify-center pt-12 bg-transparent cursor-pointer hover:scale-105 transition-transform duration-300">
        <img src={changeProfile} alt="Logo" className="w-40 h-40" />
      </div>
      <h2 className="text-lg font-bold text-[#1E1E1E] mt-4">
        {user ? `${user.first_name} ${user.last_name}` : "Loading..."}
      </h2>
      <p className="text-sm text-[#858585] mb-6">
        {user?.is_authorized ? user.user_organization : "No organization"}
      </p>

      <div className="flex flex-col gap-3 w-full pt-30">
        {NAV_ITEMS.map(({ key, label, icon, iconActive }) => (
          <button
            key={key}
            onClick={() => onTabChange(key)}
            className={`w-full py-2 px-4 flex items-center gap-2 rounded-lg text-base font-medium transition-colors duration-300 ${
              activeTab === key
                ? "bg-[#A7A7A3] text-white shadow-inner"
                : "bg-transparent text-[#1E1E1E] hover:bg-[#c4c3c1]"
            }`}
          >
            <img src={activeTab === key ? iconActive : icon} alt="" className="w-4 h-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}