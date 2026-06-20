// User CRUD endpoints, built on the shared HTTP client.
import { del, get, post, put } from "./client";

const RESOURCE = "/users/";

export const listUsers = (search = "") => get(RESOURCE, { search });

export const createUser = (payload) => post(RESOURCE, payload);

export const updateUser = (id, payload) => put(`${RESOURCE}${id}/`, payload);

export const deleteUser = (id) => del(`${RESOURCE}${id}/`);
