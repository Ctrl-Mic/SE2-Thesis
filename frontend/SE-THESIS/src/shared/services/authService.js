import axiosClient from "../api/axiosClient.api";

// *Authentication service. This is done pretty much so nothing to change here.

export const registerUser = async (userData) => {
  try {
    const response = await axiosClient.post("/auth/register", userData);
  return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }

};

export const loginUser = async (userData) => {
  try {
    const response = await axiosClient.post("/auth/login", userData);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};
export const fetchUser = async (token) => {
    try {
      const response = await axiosClient.get("/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      return response.data
    } catch (error) {
      throw error;
    }
  };

export const checkFirstTime = async () => {
  try {
    const token = localStorage.getItem("token");
    const response = await axiosClient.get("/auth/first-login", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const updateOrganization = async (organizationId) => {
  try {
    const token = localStorage.getItem("token");
    const response = await axiosClient.patch("/auth/update-organization", 
      { user_organization: organizationId },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};
