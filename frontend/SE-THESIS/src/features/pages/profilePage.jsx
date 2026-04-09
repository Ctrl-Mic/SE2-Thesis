import React, { useState } from "react";
import ProfileSidebar from "../../shared/components/layouts/profileSidebar";
import PersonalInfo from "./personalInfo";
import ApplyOrg from "./applyOrg";
import Logout from "./logout";
import { useAuth } from "../../context/authContext.jsx";

const TAB_COMPONENTS = {
  personal: PersonalInfo,
  apply: ApplyOrg,
  logout: Logout,
};



export default function ProfilePage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("personal");
  const ActiveTabComponent = TAB_COMPONENTS[activeTab];

  return (
    <section className="relative w-screen h-screen flex flex-col gap-2 bg-[#E4E3E1] px-12 py-6">
      <div className="w-full h-16 flex flex-row items-center justify-between bg-transparent">
        <h1 className="text-subtitle text-[#1E1E1E] opacity-75 font-bold">
          Intelligent Room Interaction System
        </h1>
      </div>
      <section className="relative w-full h-full flex flex-row items-start justify-start gap-10">
        <ProfileSidebar activeTab={activeTab} onTabChange={setActiveTab} user={user} />
        <div className="w-4/5 h-full aspect-video rounded-2xl shadow-outside-dropshadow flex justify-center items-center text-header overflow-auto">
          <ActiveTabComponent />
        </div>
      </section>
    </section>
  );
}