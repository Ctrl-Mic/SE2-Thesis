import { createContext, useState, useContext, useEffect } from "react";

const DeviceStateContext = createContext();

export function DeviceStateProvider({ children }) {
  const [deviceStates, setDeviceStates] = useState({});

  // Load device states from localStorage on mount
  useEffect(() => {
    const savedStates = localStorage.getItem("deviceStates");
    if (savedStates) {
      setDeviceStates(JSON.parse(savedStates));
    }
  }, []);

  // Save device states to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("deviceStates", JSON.stringify(deviceStates));
  }, [deviceStates]);
  

  const getDeviceState = (roomId) => {
    return deviceStates[roomId] || { lightsOn: false, fansOn: false };
  };

  const setDeviceState = (roomId, lightsOn, fansOn) => {
    setDeviceStates((prev) => ({
      ...prev,
      [roomId]: { lightsOn, fansOn },
    }));
  };

  const toggleLights = (roomId) => {
    const currentState = getDeviceState(roomId);
    setDeviceState(roomId, !currentState.lightsOn, currentState.fansOn);
  };

  const toggleFans = (roomId) => {
    const currentState = getDeviceState(roomId);
    setDeviceState(roomId, currentState.lightsOn, !currentState.fansOn);
  };

  return (
    <DeviceStateContext.Provider
      value={{
        deviceStates,
        getDeviceState,
        setDeviceState,
        toggleLights,
        toggleFans,
      }}
    >
      {children}
    </DeviceStateContext.Provider>
  );
}

export function useDeviceState() {
  const context = useContext(DeviceStateContext);
  if (!context) {
    throw new Error("useDeviceState must be used within DeviceStateProvider");
  }
  return context;
}
